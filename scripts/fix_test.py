content = open('tests/test_ml.py').read()
old = 'from ml.ensemble import EnsembleScorer, FeatureStore, features_to_vector, FEATURE_NAMES'
new = 'from ml.ensemble import EnsembleScorer, FeatureStore, features_to_vector, FEATURE_NAMES, INPUT_DIM'
open('tests/test_ml.py', 'w').write(content.replace(old, new))
print('Done')