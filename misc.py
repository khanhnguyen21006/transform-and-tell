import nltk
import os
import json
from tqdm import tqdm
from collections import Counter


def count_pos_tag(split, base_expt, out_path, counter):
    with open(os.path.join(base_expt, split.lower() + '.cap'), 'r') as j:
        print('reading captions ...')
        caps = j.readlines()
        with open(os.path.join(out_path, f'{split.lower()}_caption_nes.jsonl'), 'a') as f:
            for cap in tqdm(caps):
                cap = cap.strip()
                tokens = nltk.word_tokenize(cap)
                cap = nltk.Text(tokens)
                tags = nltk.pos_tag(cap)
                for _, tag in tags:
                    c.update({tag: 1})
                cap_obj = {
                    'tags': tags,
                    'length': len(tags)
                }
                f.write(f'{json.dumps(cap_obj)}\n')

    with open(os.path.join(base_expt, split.lower() + '.desc'), 'r') as j:
        descs = j.readlines()
        print('reading descriptions ...')
        with open(os.path.join(out_path, f'{split.lower()}_description_nes.jsonl'), 'a') as f:
            for desc in tqdm(descs):
                desc = desc.strip()
                tokens = nltk.word_tokenize(desc)
                desc = nltk.Text(tokens)
                tags = nltk.pos_tag(desc)
                for _, tag in tags:
                    c.update({tag: 1})
                desc_obj = {
                    'tags': tags,
                    'length': len(tags)
                }
                f.write(f'{json.dumps(desc_obj)}\n')
    print(f'{split} counter: {counter}')


if __name__ == '__main__':
    c = Counter()
    count_pos_tag('VAL', '/home/vkhanh/test/1_wit_s2s_csd/', '/home/vkhanh/test/1_wit_s2s_csd/', c)
    count_pos_tag('TEST', '/home/vkhanh/test/1_wit_s2s_csd/', '/home/vkhanh/test/1_wit_s2s_csd/', c)
    count_pos_tag('TRAIN', '/home/vkhanh/test/1_wit_s2s_csd/', '/home/vkhanh/test/1_wit_s2s_csd/', c)