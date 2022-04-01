import spacy
import json
import re
import argparse
import tqdm
from spacy.tokens import Doc
from nltk.corpus import stopwords
# from pycocoevalcap.cider.cider_scorer import CiderScorer
import os
import pickle
import string
from nltk.tokenize import sent_tokenize, word_tokenize

nlp = spacy.load("en_core_web_lg")

# https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\s*[\’\'\"]+\s*t", "", phrase)
    phrase = re.sub(r"can\s*[\’\'\"]+\s*t", "", phrase)
    # general
    phrase = re.sub(r"n\s*[\’\'\"]+\s*t", "", phrase)
    phrase = re.sub(r"\s*[\’\'\"]+\s*re", "", phrase)
    # phrase = re.sub(r"\'s", "", phrase)
    phrase = re.sub(r"\s*[\’\'\"]+\s*s", "", phrase)
    phrase = re.sub(r"\s*[\’\'\"]+\s*d", "", phrase)
    phrase = re.sub(r"\s*[\’\'\"]+\s*ll", "", phrase)
    phrase = re.sub(r"\s*[\’\'\"]+\s*t", "", phrase)
    phrase = re.sub(r"\s*[\’\'\"]+\s*ve", "", phrase)
    phrase = re.sub(r"\s*[\’\'\"]+\s*m", "", phrase)
    return phrase

def get_raw_data(image_id):
	idx = image_ids.index(image_id)
	return raw_str_caps[idx], raw_str_articles[idx]

def get_entities(doc):
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'tokens': [{'text': tok.text, 'pos': tok.pos_} for tok in ent],
        })
    return entities

def get_proper_nouns(doc):
    proper_nouns = []
    for token in doc:
        if token.pos_ == 'PROPN':
            proper_nouns.append(token.text)
    return proper_nouns

def get_cache():
	cache_path = '/media/abiten/4TB/Wiki_ECCV/good_news_entity.pkl'
	if os.path.exists(cache_path):
		print('Loading NEs ...')
		with open(cache_path, 'rb') as f:
			cache = pickle.load(f)
	else:
		print('Extracting NEs ...')
		with open('/media/abiten/4TB/Wiki_ECCV/TEST_IMAGEIDS.json', 'r') as t:
			image_ids = json.load(t)
		with open('/media/abiten/4TB/Wiki_ECCV/TEST_RAWSTRCAPS.json', 'r') as t:
			raw_str_caps = json.load(t)
		with open('/media/abiten/4TB/Wiki_ECCV/TEST_RAWSTRARTICLES.json', 'r') as t:
			raw_str_articles = json.load(t)
		cache = {}
		for img_id, cap, art in tqdm.tqdm(zip(image_ids, raw_str_caps, raw_str_articles)):
			cache.update({
					img_id: {
						'raw_caption': cap,
						'caption_NEs': [ent.text for ent in nlp(cap).ents],
						'raw_article': art,
						'article_NEs': [ent.text for ent in nlp(art).ents]
					}
				})

		if not os.path.exists(cache_path):
			with open(cache_path, 'wb') as f:
				pickle.dump(cache, f)
	return cache


def compute_pre_rec(path, tat):
	cache = get_cache()
	THRESHOLD_NE = 1
	STOPWORDS = stopwords.words('english')
	with open(path, 'r') as j:
		jsonl_content = list(j)
		cap_matches = 0
		cap_total = 0
		gen_matches = 0
		gen_total = 0
		count = 0
		TP = 0
		easy_tp = 0
		hard_tp = 0
		denom_pr_easy, denom_re_easy = 0, 0
		denom_pr_hard, denom_re_hard = 0, 0
		denom_pr, denom_re = 0, 0
		for j in tqdm.tqdm(jsonl_content):
			gen_obj = json.loads(j)
			if tat:
				i_id = (gen_obj['image_path'].split('/')[-1]).split('.')[0]
			else:
				i_id = gen_obj['image_id']

			raw_caption, raw_article, cap_NEs, article_NEs = cache[i_id]['raw_caption'], cache[i_id]['raw_article'], cache[i_id]['caption_NEs'], cache[i_id]['article_NEs']
			if raw_caption and raw_article:
				count += 1

				gen_doc = nlp(gen_obj['generation'])
				gen_NEs = [ent.text for ent in gen_doc.ents]
				# gen_tokens = [token.text for token in gen_doc]

				cap_NEs = [decontracted(c) for c in cap_NEs]
				# article_NEs = [decontracted(c) for c in article_NEs]
				gen_NEs = [decontracted(c) for c in gen_NEs]
				# gt_NEs = list(set(cap_NEs + article_NEs))

				for cap_ne in cap_NEs:
					if cap_ne in gen_obj['generation']:
						TP += 1
						if len(cap_NEs)>THRESHOLD_NE:
							hard_tp+=1
						else:
							easy_tp+=1

				if len(cap_NEs) > THRESHOLD_NE:
					denom_re_hard += len(cap_NEs)
					denom_pr_hard +=  len(gen_NEs)
				else:
					denom_re_easy += len(cap_NEs)
					denom_pr_easy += len(gen_NEs)

				denom_pr+=len(gen_NEs)
				denom_re+=len(cap_NEs)
		print("Precision: {}, Recall:{}".format(TP/denom_pr, TP/denom_re))
		print("Precision Easy: {}, Recall Easy:{}".format(easy_tp / denom_pr_easy, easy_tp / denom_re_easy))
		print("Precision Hard: {}, Recall Hard:{}".format(hard_tp / denom_pr_hard, hard_tp / denom_re_hard))
				# G = [gt_NE for gt_NE in gt_NEs if gt_NE in gen_obj['generation']]
				# G = list(set(G+gen_NEs))
				# if sorted(G) != sorted(gen_NEs):
				# 	print("INT!")
				#
				# for gen_NE in gen_NEs:
				# 	gen_NE_tokens = [token.text for token in nlp(gen_NE)]
				# 	for gen_NE_token in gen_NE_tokens:
				# 		if gen_NE_token not in G:
				# 			G.append(gen_NE_token)
				#
				# gen_matches_list = []
				# for g in G:
				# 	for cap_NE in cap_NEs:
				# 		if g in cap_NE:
				# 			gen_matches_list.append(g)
				#
				#
				# gen_matches += len(gen_matches_list)
				# gen_total += len(G)

				#cap_matches += len(cap_token_match) + len(cap_ne_match)			
				#cap_total +=  len(cap_token_match) + len(cap_nes)

		# print('total examples: ', count)
		#
		# print(f'gen_matches {gen_matches}, gen_total {gen_total}, precision : {gen_matches/gen_total}')
		#print(f'cap_matches {cap_matches}, cap_total {cap_total}, recall : {cap_matches/cap_total} ')


def remove_punc(path, tat):
	with open(path, 'r') as j:
		jsonl_content = list(j)

		out_path = os.path.join(f'./generations_remove_punctuation.jsonl')
		with open(out_path, 'a') as a:
			for j in tqdm.tqdm(jsonl_content):
				gen_obj = json.loads(j)
				if tat:
					i_id = (gen_obj['image_path'].split('/')[-1]).split('.')[0]
				else:
					i_id = gen_obj['image_id']

				caption = gen_obj['raw_caption']
				caption = [c_word for c_sent in sent_tokenize(caption) for c_word in word_tokenize(c_sent) if not re.fullmatch('[' + string.punctuation + ']+', c_word)]
				caption = ' '.join(caption).strip()
				
				generation = gen_obj['generation']
				generation = [c_word for c_sent in sent_tokenize(generation) for c_word in word_tokenize(c_sent) if not re.fullmatch('[' + string.punctuation + ']+', c_word)]
				generation = ' '.join(generation).strip()

				cap_doc = nlp(caption)
				gen_doc = nlp(generation)

				obj = {
					'caption': caption,
					'raw_caption': caption,
					'generation': generation,
					'web_url': gen_obj['web_url'] if tat else '',
					'image_path': gen_obj['image_path'] if tat else '',
					'image_id': i_id,
					'context': gen_obj['context'],
					'caption_names': get_proper_nouns(cap_doc),
					'generated_names': get_proper_nouns(gen_doc),
					'caption_entities': get_entities(cap_doc),
					'generated_entities': get_entities(gen_doc),
				}

				a.write(f'{json.dumps(obj)}\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='pre rec')
	parser.add_argument('--path', default='/media/abiten/4TB/Wiki_ECCV/generations_tat.jsonl', help='path')
	parser.add_argument('--tat', action='store_false', help='True if TAT - False if ours')
	agrs = parser.parse_args()
	path = agrs.path
	print(path)
	tat = agrs.tat
	get_cache()
	compute_pre_rec(path, tat)

	# remove_punc(path, tat)

