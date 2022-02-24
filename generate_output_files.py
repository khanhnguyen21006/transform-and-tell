from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from tqdm import tqdm
import pandas as pd
import string
import re
import os
import json
import argparse


def load_test_df(overlap_field):
	test_df = pd.read_pickle('/data2fast/users/vkhanh/data_preprocessing/wit_en_merged_processed_mod_test_df.pkl')
	print(f'test set size: {len(test_df)}, context: ' + overlap_field.split('_')[-1])
	test_easy = test_df[test_df[overlap_field] >= 0.5]
	test_hard = test_df[test_df[overlap_field] < 0.5]
	print('easy: ', len(test_easy))
	print('hard: ', len(test_hard))
	return test_df, test_easy, test_hard


def overlap(gen, b):
	gen = set(gen)
	if len(gen) == 0:
		return 0
	b = set(b)  
	overlap = gen.intersection(b)
	return len(overlap) / len(gen)

def is_correct(categorized, image_id):
	
	if str(image_id) in categorized:
		return True if categorized[str(image_id)]['type'] == 'correct' else False
	return False


def generate_files(context_field, overlap_field, expt_base, output_path, sep=False, gns=False):	

	with open(os.path.join(expt_base, 'generations.jsonl'), 'r') as j:
		jsonl_content = list(j)
		results = []		
		for jl_id, jline in enumerate(jsonl_content):
			j_content = json.loads(jline)
			results.append(j_content)
		res_df = pd.DataFrame(results)
		test_df, test_easy, test_hard = load_test_df(overlap_field)
		if sep:
			correct = []
			incorrect = []
			cor_references_coco = {u'info': {}, u'images': [], u'licenses': {}, u'type': u'captions', u'annotations': []}
			cor_hypotheses_coco = list()
			cor_test_gen = []
			incor_references_coco = {u'info': {}, u'images': [], u'licenses': {}, u'type': u'captions', u'annotations': []}
			incor_hypotheses_coco = list()
			incor_test_gen = []

		references_coco = {u'info': {}, u'images': [], u'licenses': {}, u'type': u'captions', u'annotations': []}
		hypotheses_coco = list()
		test_gen = []
		count = 0
		for index, row in tqdm(res_df.iterrows()):
			bleu_scorer = BleuScorer(n=4)
			ref_cap = row.caption
			hyp_cap = row.generation
			references_coco[u'images'].append({u'license': 3, u'file_name': str(index), u'id': index})
			references_coco[u'annotations'].append({u'image_id': index, u'id': index + 20000, u'caption': ref_cap.strip()})
			hypotheses_coco.append({u'caption': hyp_cap.strip(), u'image_id': index})
			tokenized_ref_cap = [word for word in word_tokenize(ref_cap) if not re.fullmatch('[' + string.punctuation + ']+', word)]
			tokenized_hyp_cap = [word for word in word_tokenize(hyp_cap) if not re.fullmatch('[' + string.punctuation + ']+', word)]
			tokenized_cntx = [word for word in word_tokenize(row.context) if not re.fullmatch('[' + string.punctuation + ']+', word)]
			bleu_scorer += (hyp_cap, [ref_cap])
			score, _ = bleu_scorer.compute_score(option='closest')
						
			res_map = {
				
				'context': row.context,
				'gt_caption_str': ref_cap,
				'gen_caption_str': hyp_cap,
				'gen_length': len(tokenized_hyp_cap),
				'bleu-4': score[3] * 100
			}
			if gns:
				ovl = overlap(tokenized_ref_cap, tokenized_cntx)
				res_map.update({
					'overlap': ovl,
					'type': "easy" if ovl > 0.5 else "hard",
					'gen_copied': overlap(tokenized_hyp_cap, tokenized_cntx),
				})
			else:								
				url_ser = test_df[['image_url', overlap_field, context_field, 'tokenized_cad']][(test_df.tokenized_caption.isin([tokenized_ref_cap]))]
				if len(url_ser) == 0:
					count +=1
				res_map.update({
					'image_url': url_ser.iloc[0].image_url if len(url_ser) > 0 else "",
					'type': ("easy" if url_ser.index.isin(test_easy.index)[0] else "hard") if len(url_ser) > 0 else "",
					'overlap': url_ser.iloc[0][overlap_field] if len(url_ser) > 0 else "",
					'gen_copied': overlap(tokenized_hyp_cap, url_ser.iloc[0][context_field]) if len(url_ser) > 0 else 0,
                                        'description': ' '.join(url_ser.iloc[0].tokenized_cad).strip() if len(url_ser) > 0 else ""
				})
			
			test_gen.append(res_map)
			if sep:
				categorized = json.load(open(os.path.join(output_path, 'TEST_CATEGORIZED_RET_OUTPUT_1k.json'), 'r'))
				if is_correct(categorized, row.image_id):
					correct.append(res_map)
					cor_references_coco[u'images'].append({u'license': 3, u'file_name': str(index), u'id': index})
					cor_references_coco[u'annotations'].append({u'image_id': index, u'id': index + 20000, u'caption': ref_cap.strip()})
					cor_hypotheses_coco.append({u'caption': hyp_cap.strip(), u'image_id': index})
				else:
					incor_references_coco[u'images'].append({u'license': 3, u'file_name': str(index), u'id': index})
					incor_references_coco[u'annotations'].append({u'image_id': index, u'id': index + 20000, u'caption': ref_cap.strip()})
					incor_hypotheses_coco.append({u'caption': hyp_cap.strip(), u'image_id': index})
					ret_images = test_df[test_df.index == categorized[str(row.image_id)]['ret_image']]				
					res_map.update({'false_image_url': ret_images.iloc[0].image_url if len(ret_images) > 0 else "",})
					incorrect.append(res_map)
			
			
		assert len(references_coco[u'annotations']) == len(hypotheses_coco)
		os.makedirs(output_path + '/results', exist_ok = True) 
		with open(output_path + '/results/captions_wiki_beam_5.json', 'w') as f:
			json.dump(references_coco, f)
		with open(output_path + '/results/captions_wiki_beam_5_results.json', 'w') as f:
			json.dump(hypotheses_coco, f)
		results_beam_5 = {'test_results': test_gen}
		with open(output_path + '/results/test_results_beam_5.json', 'w') as f:
			json.dump(results_beam_5, f)
		test_results_df = pd.DataFrame(test_gen)
		print('generation mean length: ', test_results_df.gen_length.mean())
		print('invalid query: ', count)
		test_results_df.to_csv(output_path + '/results/test_results.csv', index=False)

		if sep:
			assert len(cor_references_coco[u'annotations']) == len(cor_hypotheses_coco)
			assert len(incor_references_coco[u'annotations']) == len(incor_hypotheses_coco)
			with open(output_path + '/results/correct_captions_wiki_beam_5.json', 'w') as f:
				json.dump(cor_references_coco, f)
			with open(output_path + '/results/incorrect_captions_wiki_beam_5.json', 'w') as f:
				json.dump(incor_references_coco, f)	
			with open(output_path + '/results/correct_captions_wiki_beam_5_results.json', 'w') as f:
				json.dump(cor_hypotheses_coco, f)
			with open(output_path + '/results/incorrect_captions_wiki_beam_5_results.json', 'w') as f:
				json.dump(incor_hypotheses_coco, f)
			cor_results_beam_5 = {'test_results': correct}
			with open(output_path + '/results/correct_test_results_beam_5.json', 'w') as f:
				json.dump(cor_results_beam_5, f)
			cor_test_results_df = pd.DataFrame(correct)

			print('correct generation mean length: ', cor_test_results_df.gen_length.mean())
			cor_test_results_df.to_csv(output_path + '/results/correct_test_results.csv', index=False)

			incor_results_beam_5 = {'test_results': incorrect}
			with open(output_path + '/results/incorrect_test_results_beam_5.json', 'w') as f:
				json.dump(incor_results_beam_5, f)
			incor_test_results_df = pd.DataFrame(incorrect)
			print('incorrect generation mean length: ', incor_test_results_df.gen_length.mean())
			incor_test_results_df.to_csv(output_path + '/results/incorrect_test_results.csv', index=False)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='generate transform-and-tell output files for computing captioning metric scores') # expt/wit/6_transformer_weighted_roberta_test_ret_out/serialization_csd
	parser.add_argument('cntx_field', help='fieldname of the context in the raw database')  # tokenized_cad, tokenized_csd
	parser.add_argument('ovl_field', help='fieldname of the overlap degree between context and caption in the raw database') # overlap_desc, overlap_sec
	parser.add_argument('expt_base', help='path to the experiment base folder')
	parser.add_argument('out_path', help='output path')
	parser.add_argument('sep', help='separate correct and incorrect retrieval')
	parser.add_argument('gns', help='good news')
	
	agrs = parser.parse_args()
	cntx_field = agrs.cntx_field
	expt_base = agrs.expt_base
	ovl_field = agrs.ovl_field
	output_path = agrs.out_path
	sep = agrs.sep == 'True'
	gns = agrs.gns == 'True'
	
	generate_files(cntx_field, ovl_field, expt_base, output_path, sep=sep, gns=gns)
