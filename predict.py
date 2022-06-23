import pandas as pd

test_df = pd.read_csv(path/'test.csv')

test_df['inputs'] = test_df.discourse_type + sep + test_df.discourse_text

test_ds = get_dds(test_df,train=False)

preds = F.softmax(torch.Tensor(trainer.predict(test_ds).predictions)).numpy().astype(float)


submission_df = pd.read_csv(path/'sample_submission.csv')
submission_df['Ineffective'] = preds[:,0]
submission_df['Adequate'] = preds[:,1]
submission_df['Effective'] = preds[:,2]

submission_df.to_csv('submission.csv',index=False)