
def batchify(data, bsz, TEXT, device):
    # テキストをインデックスに変換
    data = TEXT.numericalize([data.examples[0].text])
    # データセットをbszサイズに分割した際のバッチ数を求める
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    # メモリ上で整列させる．これをしないとviewでエラーが出る
    data = data.view(bsz, -1).t().contiguous()  
    return data.to(device)

def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

