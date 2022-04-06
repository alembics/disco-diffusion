import json
import sys
from os import path

header_comment = '# %%\n'

def py2nb(py_str):
    # remove leading header comment
    if py_str.startswith(header_comment):
        py_str = py_str[len(header_comment):]

    cells = []
    chunks = py_str.split('\n\n%s' % header_comment)

    for chunk in chunks:
        cell_type = 'code'
        new_json = {'metadata':{}}
        if chunk.startswith('# !!'):
            new_json = json.loads("\n".join([x.strip() for x in chunk.splitlines() if '# !!' in x]).replace('# !!',''))
            chunk = "\n".join([x for x in chunk.splitlines() if '# !!' not in x])
        if chunk.startswith("'''"):
            chunk = chunk.strip("'\n")
            cell_type = 'markdown'
        elif chunk.startswith('"""'):
            chunk = chunk.strip('"\n')
            cell_type = 'markdown'

        cell = {
            'cell_type': cell_type,
            'metadata': new_json['metadata'],
            'source': chunk.splitlines(True),
        }

        if cell_type == 'code':
            cell.update({'outputs': [], 'execution_count': None})

        cells.append(cell)

    notebook = {
        'cells': cells,
        'metadata': {
            'anaconda-cloud': {},
            'accelerator': 'GPU',
            'colab': {
              'collapsed_sections': [
                'CreditsChTop',
                'TutorialTop',
                'CheckGPU',
                'InstallDeps',
                'DefMidasFns',
                'DefFns',
                'DefSecModel',
                'DefSuperRes',
                'AnimSetTop',
                'ExtraSetTop'
              ],
              'machine_shape': 'hm',
              'name': 'Disco Diffusion v5.1 [w/ Turbo]',
              'private_outputs': True,
              'provenance': [],
              'include_colab_link': True
            },
            'kernelspec': {
              'display_name': 'Python 3',
              'language': 'python',
              'name': 'python3'
            },
            'language_info': {
              'codemirror_mode': {
                'name': 'ipython',
                'version': 3
              },
              'file_extension': '.py',
              'mimetype': 'text/x-python',
              'name': 'python',
              'nbconvert_exporter': 'python',
              'pygments_lexer': 'ipython3',
              'version': '3.6.1'
            }
          },
          'nbformat': 4,
          'nbformat_minor': 4
    }

    return notebook


def convert(in_file, out_file):
    _, in_ext = path.splitext(in_file)
    _, out_ext = path.splitext(out_file)

    if in_ext == '.py' and out_ext == '.ipynb':
        with open(in_file, 'r', encoding='utf-8') as f:
            py_str = f.read()
        notebook = py2nb(py_str)
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)

    else:
        raise(Exception('Extensions must be .ipynb and .py or vice versa'))


convert('disco.py', 'Disco_Diffusion.ipynb')
