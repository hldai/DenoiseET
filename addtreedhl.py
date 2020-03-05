import argparse
import multiprocessing
import json
import spacy
from spacy import displacy
from multiprocessing import Pool

nlp = spacy.load('en')


def get_tree(sent):
    doc = nlp(sent)
    tree = []
    for token in doc:
        d = {}
        d['text'] = token.text
        d['lemma'] = token.lemma_
        d['pos'] = token.pos_
        d['tag'] = token.tag_
        d['dep'] = token.dep_
        d['shape'] = token.shape_
        d['is_alpha'] = token.is_alpha
        d['is_stop'] = token.is_stop
        tree.append(d)
    return tree


def add_tree(x):
    # print('processing: ' + x['annot_id'])
    x['mention_span_tree'] = get_tree(x['mention_span'].strip())
    return x


def __process_data(filename, output_file):
    block_size = 100000
    cores_to_use = 20

    idx = 0
    f = open(filename, encoding='utf-8')
    fout = open(output_file, 'w', encoding='utf-8', newline='\n')
    while True:
        print(f'processing {idx} ...')
        mentions = list()
        for line in f:
            mentions.append(json.loads(line))
            if len(mentions) == block_size:
                break

        if len(mentions) == 0:
            break

        p = Pool(cores_to_use)
        data_w_tree = p.map(add_tree, mentions)
        for x in data_w_tree:
            fout.write('{}\n'.format(json.dumps(x)))

        if len(mentions) < block_size:
            break
        idx += 1
    f.close()
    fout.close()
    print('==> saved data to: ' + output_file)


if __name__ == '__main__':
    # file_from, file_to = config.read_from, config.save_to
    # file_from = '/home/data/hldai/ultrafine/uf_data/train/open_train_00.json'
    # file_to = '/home/data/hldai/ultrafine/ld_data/train/open_train_tree_00.json'
    file_from = '/home/data/hldai/ultrafine/uf_data/train/el_train.json'
    file_to = '/home/data/hldai/ultrafine/ld_data/train_full/el_train_full_tree.json'
    __process_data(file_from, file_to)
