from os.path import join
from platform import platform

if platform().startswith('Windows'):
    ALT_DATA_PATH = 'd:/data/ultrafine/ld_data'
else:
    ALT_DATA_PATH = '/home/data/hldai/ultrafine/ld_data'

DEF_FILE = join(ALT_DATA_PATH, 'ontology/types_definition.txt')
TYPES_FILE = join(ALT_DATA_PATH, 'ontology/types.txt')
ONTO_TYPES_FILE = join(ALT_DATA_PATH, 'ontology/onto_ontology.txt')
CHAR_VOCAB_FILE = join(ALT_DATA_PATH, 'ontology/char_vocab.english.txt')

TRAIN_M_TREE_FILE = join(ALT_DATA_PATH, 'crowd/train_m_tree.json')
