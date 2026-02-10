#!/bin/sh
set -e
export SPACY_DATA=/data
# Copy models from image site-packages to /data (host volume) on first run. Uses Python so we get exact paths spacy uses.
if [ ! -d /data/en_core_web_md ]; then
  uv run python -c "
import os, shutil
models = ['en_core_web_md', 'fr_core_news_md', 'de_core_news_md', 'nl_core_news_md', 'ru_core_news_md', 'xx_ent_wiki_sm']
for name in models:
    dest = os.path.join('/data', name)
    if os.path.isdir(dest):
        continue
    try:
        import spacy
        nlp = spacy.load(name)
        shutil.copytree(nlp.path, dest)
    except Exception as e:
        print(f'Copy {name}: {e}', flush=True)
        raise
"
fi
exec "$@"
