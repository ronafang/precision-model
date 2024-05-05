import numpy as np
from scipy.io import wavfile
import torch.nn.functional as F
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from scipy.signal import windows

def cosine_similarity_loss(output, target):
    # similarity score out of 1, but the loss version so 1 - it
    
    # normalize so its moreso a structural similarity
    output_norm = F.normalize(output, p=2, dim=0)
    target_norm = F.normalize(target, p=2, dim=0)
    cos_sim = torch.dot(output_norm, target_norm)
    loss = 1 - cos_sim
    return loss

def pos_embeddings(max_position=4096):
    # adding some sinusoidal relationships based on index that the model can hopefully grasp on
    positions = torch.arange(max_position).reshape(max_position, 1)
    div_term = torch.exp(torch.arange(0, 1, 2) * -(torch.log(torch.tensor(10000.0)))).requires_grad_(True)
    embeddings = torch.sin(positions * div_term).requires_grad_(True)
    return embeddings.squeeze()

class AudioInput:
    def __init__(self, path):
        sample_rate, data = wavfile.read(path)
        self.length = len(data)
        self.cur = 0
        self.sample_rate = sample_rate
        self.data = data.mean(axis=1)
        
        def raw_chunks():
            # separating audio to 4096 samples, every 2048 samples
            # so we can add them with hamming window w overlap so its smoother, later
            output = [np.copy(self.data[:4096])]
            self.cur += 2048

            while self.cur + 4096 < self.length:
                self.cur += 2048
                output.append(np.copy(self.data[self.cur-4096:self.cur]))

            vals = np.zeros(4096)
            vals[0:self.length-self.cur] = np.copy(self.data[self.cur:])
            self.cur = 0
            output.append(vals)
            return output

        chunks = raw_chunks()
        
        # the absolute value of the fft results such that it takes in account of the complex vals
        fft_results = [torch.fft.fft(torch.from_numpy(chunk).requires_grad_(True)).abs() for chunk in chunks]
        
        # the actual pure fft values
        self.ffts = [torch.fft.fft(torch.from_numpy(chunk).requires_grad_(True)) for chunk in chunks]
        
        POS_EMBS = pos_embeddings()
        
        # just combining positional embeddings with the fft results
        fft_results = [fft_result + POS_EMBS for fft_result in fft_results]
        
        # cumulative sums so that we can easily do range sum queries
        self.sums = [torch.cumsum(fft_result, dim=0) for fft_result in fft_results]

class InstrEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model=1, nhead=1, dim_feedforward=512, dropout=0.1, activation="relu"):
        super(InstrEncoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.d_model = d_model
        self.nhead = nhead
        
        # the main attention values we'll be using
        self.zoom_attn = nn.MultiheadAttention(d_model, nhead, dtype=torch.float32)
        
        # arbitrary feed forward network to do things 
        # we'll use this after we do some operations with the zoom attention scores
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.PReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # normalization functions 
        self.bn1 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(d_model)

        # random weights
        nn.init.normal_(self.zoom_attn.in_proj_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.zoom_attn.out_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.zoom_attn.in_proj_bias, mean=0.0, std=0.01)
        nn.init.normal_(self.zoom_attn.out_proj.bias, mean=0.0, std=0.01)

    def unnormalize(self, bn, x):
        
        # we normalized our FFT values but we want to interpret our FFT values as FFT values
        # so we un-normalize by the weights and biases that our normalization used

        mean = bn.running_mean
        std = torch.sqrt(bn.running_var + bn.eps)
        weight = bn.weight
        bias = bn.bias
        x = x * std[:, None] / weight[:, None] + (mean[:, None] - bias[:, None]) / weight[:, None]

        return x

    def forward(self, src, fft, **kwargs):
        # src is 2048 numbers, first 2048-256 are historical / context window, final 256 is current
        
        # normalize the src
        srcnorm = self.bn1(src)

        src2 = srcnorm[-256:, :] # extract the current chunk
        
        # the historical chunk, not normalized

        src1 = src[:(2048-256),:]
        
        # get the attention scores given the normalized source
        _, zoom_weights = self.zoom_attn(srcnorm, srcnorm, srcnorm)
        
        # but we only want the zoom_weights for the 256 current values
        # but we passed everything so it can look at it and consider 
        zoom_weights = zoom_weights[(2048-256):,(2048-256):]
        
        # because we are only taking a 256 segment out of 2048, we softmax it
        # so the probabilities add up to 1 again just in our 256 section
        probabilities = nn.functional.softmax(zoom_weights, dim=1)

        # we sum the attention given to i for every 0 ... 256 
        # then softmax it back such that the attentions add up to 1
        # so that we interpret this now 1d array of 256 len of how much attention for each
        final_probabilities = nn.functional.softmax(probabilities.sum(dim=1), dim=0)

        # we interpret the probabilities/attentions into indices in our FFT
        inds = self.indices(final_probabilities)
        
        # check: inds should be considering src and fft in gradients

        # then we apply these indices to our current chunk (src2)
        src3 = self.apply_custom_attention(src2, inds, fft)

        # important to note that the format of the above isn't normalized
        # the input to get weights is normalized, but here we are back to values of original scale
        
        # so we cat the non normalized src1 ([:2048-256]) and our new src3 values
        src4 = torch.cat((src1, src3), dim=0)
        
        # but we normalize it now again
        srcnorm2 = self.bn2(src4)
        
        # so that we can apply some operations to it such that the values change
        src_transformed = self.ffn(srcnorm2)
        
        # and unnormalize it so that we get it back to original scale
        src_unnorm = self.unnormalize(self.bn2, src_transformed)
        
        # return the result and the inds which they represent
        return src_unnorm, inds
        

    def indices(self, attention_values):
        # setting it such that it tracks gradients so it can learn
        attention_values.requires_grad_(True)
        
        # we interpret attention as the (1-attention) of how much to zoom in
        # i.e 0.5 attention = range sum across 2048 units of FFT results
        # 1/4095 attention = range sum across 1 unit of FFT results

        inds = torch.cumsum(attention_values, dim=0)*4095

        inds.requires_grad_(True)
        return inds

    def apply_custom_attention(self, src2, inds, fft):
        
        def fft_interp(i):
            # just linear interpolation between 2 values
            if i>4095:
                i -= 1
            start = fft[i.floor().long()].clone()
            end = fft[i.ceil().long()].clone()
            frac = i - i.floor()
            if frac == 0:
                # such that it's still differentiable (not at node)
                if i > 0:
                    return fft_interp(i-0.001)
                else:
                    return fft_interp(i+0.001)
            slope = end-start
            res = start+frac*slope
            return res
            
        new_val = torch.zeros((inds.size(0), 1), dtype=torch.float32)
        
        # the zeroth one doesnt have a prior index to subtract from, the sum is from start anyways
        new_val[0] = fft_interp(inds[0]).unsqueeze(0)

        for i in range(1, inds.size(0)):
            # new range sum queries
            new_val[i] = (fft_interp(inds[i]) - fft_interp(inds[i - 1])).unsqueeze(0)
        
        # this is just so that we still get half the weights of src2
        # ... not sure if its good xd
        return src2/2 + new_val - src2.detach()/2
        
        # return new_val

        

class InstrEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers):
        super(InstrEncoder, self).__init__(encoder_layer, num_layers)
    
    def forward(self, src, fft):
        output = src
        for model in self.layers:
            output, positions = model(output, fft=fft)
        return output, positions
    
class InstrModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(InstrModel, self).__init__()
        self.d_model = d_model
        self.encoder_layer = InstrEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = InstrEncoder(self.encoder_layer, num_layers=num_layers)    
        
    def forward(self, src, fft):
        output = self.transformer_encoder(src, fft)
        return output


def default_input_from_fft(fft):
    # this is the already cumsummed fft
    # just takes even segments and positioning
    positions = torch.linspace(0, 4095, 256, dtype=torch.int64)
    step = 4096 // 256
    # its wrapped in a list each (unsqueeze) because we can also add more parameters here later
    # im just only using the val+pos rn
    # also because the attention mechanism gives scores per thing= [feature1, feature2, featuren...]
    # so this is just feature1 and nothing else
    values = [(fft[pos].clone() - fft[max(0, pos - step)].clone()).unsqueeze(0) if pos - step >= 0 else fft[pos].clone().unsqueeze(0) for pos in positions]
    return torch.stack(values)


def reconstruct_fft(positions, new_vals, fft, sums):

    def sums_interp(i):
        # just linear interpolation between 2 values
        start = sums[i.floor().long()].clone()
        end = sums[i.ceil().long()].clone()
        frac = i - i.floor()
        if frac == 0:
            # such that it's still differentiable (not at node)
            if i > 0:
                return sums_interp(i-0.001)
            else:
                return sums_interp(i+0.001)
        slope = end-start
        res = start+frac*slope
        return res
    reconstructed_fft = torch.zeros_like(fft)
    x = 1
    reconstructed_fft[0] = fft[0].clone() * new_vals[0]/sums_interp(positions[0])
    for i in range(1,len(fft)):
        if i > positions[x] and x < 255:
            x += 1
        if sums_interp(positions[x])-sums_interp(positions[x-1]) == 0:
            reconstructed_fft[i] = 0
        else:
            reconstructed_fft[i] = fft[i].clone() * new_vals[x]/(sums_interp(positions[x])-sums_interp(positions[x-1]))
    
    return reconstructed_fft


class AudioDataset(Dataset):
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.split = split
        self.song_folders = sorted(os.listdir(os.path.join(data_dir, split)))

    def __len__(self):
        return len(self.song_folders)

    def __getitem__(self, index):
        song_folder = self.song_folders[index]
        song_path = os.path.join(self.data_dir, self.split, song_folder)

        input_path = os.path.join(song_path, 'mixture.wav')
        output_path = os.path.join(song_path, 'accompaniment.wav')

        input_audio = AudioInput(input_path)
        output_audio = AudioInput(output_path)

        input_chunks = input_audio.sums
        input_ffts = input_audio.ffts
        output_chunks = output_audio.ffts

        segments = []
        for i in range(len(input_chunks) - 7):
            input_segment = input_chunks[i:i+8]
            input_fft = input_ffts[i+7]

            output_segment = output_chunks[i+7]
            segments.append((input_segment, input_fft, output_segment))

        return segments

def collate_fn(batch):
    input_segments = []
    output_segments = []
    for segments in batch:
        for input_segment, input_fft, output_segment in segments:
            
            input_data = torch.zeros((2048,1), dtype=torch.float32)
            
            for i in range(8):
                input_data[i*256:i*256+256] = default_input_from_fft(input_segment[i])            
            # only keep fft of last one
            input_segments.append((input_data.requires_grad_(), input_fft, input_segment[-1]))
            output_segments.append(output_segment)
    
    return input_segments, output_segments

data_dir = 'MUSDB'
train_dataset = AudioDataset(data_dir, 'train')
test_dataset = AudioDataset(data_dir, 'test')

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, pin_memory=True)

torch.save(train_dataloader, 'train_dataloader.pth')
torch.save(test_dataloader, 'test_dataloader.pth')


def reconstruct_audio(model, input_audio_path, output_audio_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_audio = AudioInput(input_audio_path)
    input_chunks = input_audio.sums
    input_ffts = input_audio.ffts
    reconstructed_audio = np.zeros(input_audio.length)

    window_size = 4096
    hop_size = 2048

    # hamming window, so when we add the overlapping chunks add up to 100%
    window = windows.hamming(window_size)

    for i in range(len(input_chunks) - 7):
        input_segment = input_chunks[i:i+8]
        input_fft = input_ffts[i+7]

        input_data = torch.zeros((2048, 1), dtype=torch.float32)
        
        # input-ify it in chunks of 8
        # tbh i should keep a queue of history of 7 to not recalculate but its ok
        for j in range(8):
            input_data[j*256:j*256+256] = default_input_from_fft(input_segment[j])

        input_data = input_data.to(device)
        input_fft = input_fft.to(device)

        with torch.no_grad():
            model_output, positions = model(input_data, input_fft)
            reconstructed_fft = reconstruct_fft(positions, model_output[-256:], input_fft, input_segment[7])

        reconstructed_chunk = torch.fft.irfft(reconstructed_fft)

        if len(reconstructed_chunk) < window_size:
            reconstructed_chunk = torch.nn.functional.pad(reconstructed_chunk, (0, window_size - len(reconstructed_chunk)))
        else:
            reconstructed_chunk = reconstructed_chunk[:window_size]

        # apply the window
        reconstructed_chunk = reconstructed_chunk.cpu().numpy() * window

        # add them up
        start = i * hop_size
        end = start + window_size

        # can just contiuously add bc we have hamming window w our hop size
        reconstructed_audio[start:end] += reconstructed_chunk

    # normalized for file output
    reconstructed_audio /= np.max(np.abs(reconstructed_audio))

    wavfile.write(output_audio_path, input_audio.sample_rate, reconstructed_audio)
    

def train_model(d_model, nhead, num_layers, learning_rate, num_epochs, train_dataloader_path, test_dataloader_path, save_model_every_n_batches=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InstrModel(d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)
    train_dataloader = torch.load(train_dataloader_path)
    test_dataloader = torch.load(test_dataloader_path)

    criterion = cosine_similarity_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        count = 0
        for i, (input_segments, output_segments) in enumerate(train_dataloader):

            batch_loss = 0
            count = 0
            for input_data, output_data in zip(input_segments, output_segments):

                count += 1
                if count > 64:
                    break
                input_vals, input_ffts, input_sums = input_data
                output_fft = output_data

                input_vals = input_vals.to(device)
                input_sums = input_sums.to(device)
                input_ffts = input_ffts.to(device)
                output_fft = output_fft.to(device)
                optimizer.zero_grad()

                model_output, positions = model(input_vals, input_sums)
                reconstructed_fft = reconstruct_fft(positions, model_output[-256:], input_ffts, input_sums)

                loss = criterion(reconstructed_fft.real, output_fft.real)
                loss.backward(retain_graph=True)
                optimizer.step()
                batch_loss += loss.item()

            batch_loss /= min(64, len(input_segments))
            running_loss += batch_loss

            if (i + 1) % save_model_every_n_batches == 0:
                reconstruct_audio(model, "mixture.wav", f"output{i}.wav")
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_dataloader)}], Loss: {running_loss / save_model_every_n_batches:.4f}')
                running_loss = 0.0

                save_path = f'./model_checkpoint_batch_{i + 1}_epoch_{epoch + 1}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'batch_num': i + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': batch_loss,
                }, save_path)
                print(f"Checkpoint saved: {save_path}")

    print('Training finished.')

def generate_audio_from_checkpoint(checkpoint_path, input_audio_path, output_audio_path, d_model, nhead, num_layers):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InstrModel(d_model=d_model, nhead=nhead, num_layers=num_layers)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    reconstruct_audio(model.to(device), input_audio_path, output_audio_path)

def evaluate_model(checkpoint_path, test_dataloader_path, device):
    d_model = 1
    nhead = 1
    num_layers = 1
    model = InstrModel(d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    test_dataloader = torch.load(test_dataloader_path)

    total_loss = 0
    count = 0
    with torch.no_grad():
        for i, (input_segments, output_segments) in enumerate(test_dataloader):
            if count % 50:
                print(count)
            if count > 500:
                break
            for input_data, output_data in zip(input_segments, output_segments):
                if count % 50 == 1:
                    print(count, total_loss/count)
                if count > 500:
                    break
                input_vals, input_ffts, input_sums = input_data
                output_fft = output_data
                
                input_vals = input_vals.to(device)
                input_sums = input_sums.to(device)
                input_ffts = input_ffts.to(device)
                output_fft = output_fft.to(device)
                
                model_output, positions = model(input_vals, input_sums)
                
                reconstructed_fft = reconstruct_fft(positions, model_output[-256:], input_ffts, input_sums)
                
                loss = cosine_similarity_loss(reconstructed_fft.real, output_fft.real)
                total_loss += loss.item()
                count += 1
    average_loss = total_loss / count
    print(f'Average loss on the test set: {average_loss}')
    return average_loss


# generate_audio_from_checkpoint("model_checkpoint_batch_12_epoch_9.pth", "ocanada.wav", "ocanada_inst.wav", 1, 1, 1)