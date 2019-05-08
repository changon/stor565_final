import pandas as pd
data=pd.read_csv('data_with_embeddings_sg.csv')
former =[]
for descrip in data['job-title']:
    if "Current" in descrip:
        former.append("current")
    else:
        former.append("former")

data['formerOrCurrent']=former

data.to_csv('data_with_embeddings_sg.csv', index=False)
