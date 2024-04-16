#!/usr/bin/python3

from os import walk
from os.path import join, exists, splitext
from absl import flags, app
from tqdm import tqdm
import json
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredFileLoader, UnstructuredMarkdownLoader

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to directory containing pdfs')
  flags.DEFINE_string('output_json', default = 'dataset.json', help = 'path to output json file')

def main(unused_argv):
  with open(FLAGS.output_json, 'w') as output:
    for root, dirs, files in tqdm(walk(FLAGS.input_dir)):
      for f in files:
        stem, ext = splitext(f)
        loader_types = {'.md': UnstructuredMarkdownLoader,
                        '.txt': UnstructuredFileLoader,
                        '.pdf': UnstructuredPDFLoader}
        loader = loader_types[ext](join(root, f), mode = "single", strategy = "fast")
        for s in loader.load():
          output.write('%s\n' % json.dumps({'text': s}))

if __name__ == "__main__":
  add_options()
  app.run(main)

