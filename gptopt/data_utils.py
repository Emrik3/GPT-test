import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from functools import partial

magic_number = 20250401         # used in the header of saved binary files

def tokenize(doc, enc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    eot = enc._special_tokens['<|endoftext|>'] # end of text token
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename, toks):
    """ 
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = magic_number # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # construct the tokens numpy array, if not already
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        # validate that no token exceeds a uint16
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())


def process_and_save_docs(
    dataset, filepath, encoding, shard_size=int(1e8), nprocs=1, max_tokens=None
):
    # tokenize all documents and write output shards, each of shard_size tokens
    # last shard has remainder

    if nprocs == 0:
        nprocs = max(1, os.cpu_count() - 2)
    print("Number of processes used  : ", nprocs)

    tokenizer = partial(tokenize, enc=encoding)

    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        total_tokens = 0
        progress_bar = None

        for tokens in pool.imap(tokenizer, dataset, chunksize=16):

            # Stop once max_tokens has been reached
            if max_tokens is not None:
                remaining = max_tokens - total_tokens
                if remaining <= 0:
                    break
                tokens = tokens[:remaining]

            # if there is space left, add tokens to current shard
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                total_tokens += len(tokens)

                if progress_bar is None:
                    progress_bar = tqdm(
                        total=shard_size,
                        unit="tokens",
                        desc=f"Shard {shard_index}"
                    )
                progress_bar.update(len(tokens))

            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(
                    filepath, f"{split}_{shard_index:06d}.bin"
                )

                remainder = shard_size - token_count
                progress_bar.update(remainder)

                all_tokens_np[
                    token_count:token_count+remainder
                ] = tokens[:remainder]

                write_datafile(filename, all_tokens_np)

                shard_index += 1
                total_tokens += remainder
                progress_bar = None

                # populate the next shard with leftovers
                leftover = tokens[remainder:]
                all_tokens_np[0:len(leftover)] = leftover
                token_count = len(leftover)
                total_tokens += len(leftover)

            if max_tokens is not None and total_tokens >= max_tokens:
                break

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                filepath, f"{split}_{shard_index:06d}.bin"
            )
            write_datafile(filename, all_tokens_np[:token_count])