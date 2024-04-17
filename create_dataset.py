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
  flags.DEFINE_enum('target', default = 'pretrain', enum_values = {'pretrain', 'sft'}, help = 'which target the dataset generator for')
  flags.DEFINE_enum('format', default = 'conv', enum_values = {'conv', 'instr'}, help = 'format of dataset used in SFT')

def main(unused_argv):
  if FLAGS.target == 'pretrain':
    with open(FLAGS.output_json, 'w') as output:
      for root, dirs, files in tqdm(walk(FLAGS.input_dir)):
        for f in files:
          stem, ext = splitext(f)
          loader_types = {'.md': UnstructuredMarkdownLoader,
                          '.txt': UnstructuredFileLoader,
                          '.pdf': UnstructuredPDFLoader}
          loader = loader_types[ext](join(root, f), mode = "single", strategy = "fast")
          for s in loader.load():
            output.write('%s\n' % json.dumps({'text': s.page_content, 'source': s.metadata['source'], 'type': s.type}))
  elif FLAGS.target == 'sft':
    with open(FLAGS.output_json, 'w') as output:
      for root, dirs, files in tqdm(walk(FLAGS.input_dir)):
        for f in files:
          stem, ext = splitext(f)
          if ext != '.json': continue
          with open(join(root, f), 'r') as input:
            lines = [line.strip() for line in input.readlines() if line.strip() != '']
            if FLAGS.format == 'conv':
              assert len(lines) == 3
              roles = ['system', 'user', 'assistant']
              sample = {"message": [{"role": role, "content": line} for role, line in zip(roles, lines)]}
              output.write(json.dumps(sample) + '\n')
            elif FLAGS.format == 'instr':
              assert len(lines) == 2
              sample = {"prompt": lines[0], "completion": lines[1]}
              output.write(json.dumps(sample) + '\n')

if __name__ == "__main__":
  add_options()
  app.run(main)

