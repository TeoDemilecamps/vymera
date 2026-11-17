"""Builds the site's index.html (repo root) by injecting the trained model and VIX data from ../ml/outputs."""
import json, os
here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, '..', '..', 'ml', 'outputs')
t = open(os.path.join(here, 'app.template.html')).read()
t = t.replace('/*__MODEL__*/null', open(os.path.join(out, 'model_weights.json')).read().strip())
t = t.replace('/*__VIX__*/null', json.dumps(json.load(open(os.path.join(out, 'vix_recent.json'))), separators=(',', ':')))
open(os.path.join(here, '..', '..', 'index.html'), 'w').write(t)
print('built index.html', len(t)//1024, 'KB')
