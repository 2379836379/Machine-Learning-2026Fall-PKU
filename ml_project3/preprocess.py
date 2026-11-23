import argparse
from pathlib import Path
import torch
from torch.utils.data import TensorDataset
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(items, **kwargs):
        return items
from module import Tokenizer
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_dataset(fdir: Path, tokenizer: Tokenizer, max_seq_len: int):
    with (fdir / "in.txt").open(encoding="utf-8") as inputs, (fdir / "out.txt").open(encoding="utf-8") as targets:
        total = sum(1 for _ in inputs)
        inputs.seek(0)
        input_ids = torch.full((total, max_seq_len), tokenizer.pad_id, dtype=torch.long)
        target_ids = torch.full((total, max_seq_len + 1), tokenizer.pad_id, dtype=torch.long)
        decoder_input_ids = torch.full_like(target_ids, tokenizer.pad_id)
        masks = torch.ones((total, max_seq_len), dtype=torch.bool)
        lens = torch.empty(total, dtype=torch.long)

        pairs = zip(inputs, targets, strict=True)
        for i, (source, target) in enumerate(tqdm(pairs, total=total, desc=f"creating {fdir.name} tensors")):
            src = tokenizer.convert_tokens_to_ids(source.split()[:max_seq_len])
            tgt = tokenizer.convert_tokens_to_ids(target.split()[:max_seq_len])
            input_ids[i, :len(src)] = torch.tensor(src, dtype=torch.long)
            masks[i, :len(src)] = False
            lens[i] = len(src)
            target_tokens = torch.tensor(tgt, dtype=torch.long)
            target_ids[i, :len(tgt)] = target_tokens
            target_ids[i, len(tgt)] = tokenizer.eos_id
            decoder_input_ids[i, 0] = tokenizer.bos_id
            decoder_input_ids[i, 1:len(tgt) + 1] = target_tokens

    return TensorDataset(input_ids, masks, lens, decoder_input_ids, target_ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default='couplet', type=str)
    parser.add_argument("--output", default='dataset', type=str)
    parser.add_argument("--max_seq_len", default=32, type=int)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    vocab_file = input_dir / "vocabs"

    logger.info("creating tokenizer...")
    tokenizer = Tokenizer()
    tokenizer.build(vocab_file)

    logger.info("creating dataset...")
    train_dataset = create_dataset(
        input_dir / "train", tokenizer, args.max_seq_len)
    test_dataset = create_dataset(
        input_dir / "test", tokenizer, args.max_seq_len)

    logger.info("saving dataset...")
    tokenizer.save_pretrained(output_dir / "vocab.pkl")
    torch.save(train_dataset, output_dir / "train.pkl")
    torch.save(test_dataset, output_dir / "test.pkl")
