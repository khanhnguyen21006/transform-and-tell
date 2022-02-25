import spacy
import json
import re
import argparse
import tqdm
from spacy.tokens import Doc
from pycocoevalcap.cider.cider_scorer import CiderScorer


nlp = spacy.load("en_core_web_lg")
with open('/data2fast/users/vkhanh/data_goodnews/TEST_IMAGEIDS.json', 'r') as t:
	image_ids = json.load(t)
with open('/data2fast/users/vkhanh/data_goodnews/TEST_RAWSTRCAPS.json', 'r') as t:
	raw_str_caps = json.load(t)

def contain_entity(entities, target):
    for ent in entities:
        if ent['text'] == target['text'] and ent['label'] == target['label']:
            return True
    return False

def get_entities(doc):
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'tokens': [{'text': tok.text, 'pos': tok.pos_} for tok in ent],
        })
    return entities

def sample(i):
	our = json.load(open('/home/vkhanh/Desktop/generations_short.json', 'r'))
	tat  = json.load(open('/home/vkhanh/Downloads/generations_short.json', 'r'))

	our_cap_doc = Doc(nlp.vocab).from_bytes(nlp(our[i]['caption']).to_bytes())
	tat_cap_doc = Doc(nlp.vocab).from_bytes(nlp(tat[i]['caption']).to_bytes())

	our_gen_doc = Doc(nlp.vocab).from_bytes(nlp(our[i]['generation']).to_bytes())
	tat_gen_doc = Doc(nlp.vocab).from_bytes(nlp(tat[i]['generation']).to_bytes())

	print('Our cap: ', our[i]['caption'])
	print('Our generation: ', our[i]['generation'])	
	
	print('TAT cap: ', tat[i]['caption'])
	print('TAT generation: ', tat[i]['generation'])

	print('Our cap doc: ', our_cap_doc.ents)
	print('Our gen doc: ', our_gen_doc.ents)
	print('TAT cap doc: ', tat_cap_doc.ents)
	print('TAT gen doc: ', tat_gen_doc.ents)

	our_cap_entities = get_entities(our_cap_doc)
	tat_cap_entities = get_entities(tat_cap_doc)
	our_gen_entities = get_entities(our_gen_doc)
	tat_gen_entities = get_entities(tat_gen_doc)

	our_cap_matches = 0
	tat_cap_matches = 0
	our_gen_matches = 0
	tat_gen_matches = 0

	for ent in our_gen_entities:
		if contain_entity(our_cap_entities, ent):
			our_gen_matches += 1

	for ent in our_cap_entities:
		if contain_entity(our_gen_entities, ent):
			our_cap_matches += 1

	for ent in tat_gen_entities:
		if contain_entity(tat_cap_entities, ent):
			tat_gen_matches += 1

	for ent in tat_cap_entities:
		if contain_entity(tat_gen_entities, ent):
			tat_cap_matches += 1

	our_caption = re.sub(r'[^\w\s]', '', our[i]['caption'])
	our_generation = re.sub(r'[^\w\s]', '', our[i]['generation'])
	tat_caption = re.sub(r'[^\w\s]', '', tat[i]['caption'])
	tat_generation = re.sub(r'[^\w\s]', '', tat[i]['generation'])

	our_cider_scorer = CiderScorer(n=4, sigma=6.0)
	tat_cider_scorer = CiderScorer(n=4, sigma=6.0)
	our_cider_scorer += (our_caption, [our_generation])
	tat_cider_scorer += (tat_generation, [tat_caption])

	print('our gen matches: ', our_gen_matches)
	print('our cap matches: ', our_cap_matches)
	print('tat gen matches: ', tat_gen_matches)
	print('tat cap matches: ', tat_cap_matches)
	print('our cider: ', our_cider_scorer.compute_score())
	print('tat cider: ', tat_cider_scorer.compute_score())

def get_raw_caption(image_id):
	return raw_str_caps[image_ids.index(image_id)]


def compute_pre_rec(path, image_path):

	with open(path, 'r') as j:
		jsonl_content = list(j)
		cap_matches = 0
		cap_total = 0
		gen_matches = 0
		gen_total = 0
		count = 0
		for jc in tqdm.tqdm(jsonl_content):
			jc_obj = json.loads(jc)
			if image_path:
				i_id = (jc_obj['image_path'].split('/')[-1]).split('.')[0]
			else:
				i_id = jc_obj['image_id']
			raw_caption = get_raw_caption(i_id)
			if raw_caption:
				count += 1
				cap_doc = nlp(raw_caption)	
				gen_doc = nlp(jc_obj['generation'])	

				cap_split = [token.text for token in cap_doc]
				cap_nes = [ent.text for ent in cap_doc.ents]
				gen_split = [token.text for token in gen_doc]
				gen_nes = [ent.text for ent in gen_doc.ents]

				match_gen_toks = []
				match_gen_nes = []
				match_cap_toks = []
				match_cap_nes = []

				for ne in gen_nes:
					for ent in cap_doc.ents:
						if ne == ent.text:
							match_gen_nes.append(ne)
				for tok in gen_split:
					for ent in cap_doc.ents:
						if tok == ent.text and tok not in match_gen_nes:
							match_gen_toks.append(tok)
				
				for ne in cap_nes:
					for ent in gen_doc.ents:
						if ne == ent.text:
							match_cap_nes.append(ne)

				for tok in cap_split:
					for ent in gen_doc.ents:
						if tok == ent.text and tok not in match_cap_nes:
							match_cap_toks.append(tok)
				

				cap_matches += len(match_cap_toks) + len(match_cap_nes)			
				cap_total +=  len(match_cap_toks) + len(cap_nes)

				gen_matches += len(match_gen_toks) + len(match_gen_nes)
				gen_total += len(match_gen_toks) + len(gen_nes)

		print('total length: ', count)

		print('precision :', gen_matches/gen_total)
		print('recall :', cap_matches/cap_total)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='pred rec')
	parser.add_argument('path', help='path')
	parser.add_argument('image_path', help='image_path')	
	agrs = parser.parse_args()
	path = agrs.path
	image_path = agrs.image_path == 'True'
	
	compute_pre_rec(path, image_path)


