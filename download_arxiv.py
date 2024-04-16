#!/usr/bin/python3

from absl import flags, app
from shutil import rmtree
from os import mkdir
from os.path import exists, join
from paperscraper.load_dumps import QUERY_FN_DICT
from paperscraper.pdf import save_pdf_from_dump

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_txt', default = None, help = 'path to key word file')
  flags.DEFINE_string('output_dir', default = 'papers', help = 'path to output directory')

def main(unused_argv):
  if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
  mkdir(FLAGS.output_dir)
  with open(FLAGS.input_txt, 'r') as f:
    query = [line for line in f.readlines()]
  for key in QUERY_FN_DICT.keys():
    QUERY_FN_DICT[key](query, output_filepath = '%s_result.jsonl' % key)
    if not exists(join(FLAGS.output_dir, key)): mkdir(join(FLAGS.output_dir, key))
    save_pdf_from_dump('%s_result.jsonl' % key, pdf_path = join(FLAGS.output_dir, key), key_to_save = 'doi')

if __name__ == "__main__":
  add_options()
  app.run(main)

