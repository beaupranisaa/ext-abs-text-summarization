import os
import pandas as pd
import argparse

def combine(args, mode = "train_set"):

    if args.approach in ['head-only','tail-only','head+tail0.2', 'head+tail0.5']:
        if args.shuffle == True:
            path = f'''../extracted/truncation/{args.approach}/shuffled/{mode}_seed{args.seed}/'''
        else:
            path = f'''../extracted/truncation/{args.approach}/unshuffled/{mode}'''
    else:
        if args.shuffle == True:
            path = f'''../extracted/extractive/{args.approach}/shuffled/{mode}_seed{args.seed}/'''
        else: 
            path = f'''../extracted/extractive/{args.approach}/unshuffled/{mode}'''    
    
    print("PATH: ", path)

    content = os.listdir(path) 
    extracted_ = [path for path in content if ".csv" in path]
    final = {"Sample ids": [], "Document": [], "Shortened Document": [], "Summary": [], "Document length": [] , "Shortened Document length": []} 
    
    for i in range(len(extracted_)):
        df = pd.read_csv(os.path.join(path, f"""step{i}.csv"""))
        final["Sample ids"].extend(df["Sample ids"])
        final["Document"].extend(df["Document"])
        final["Shortened Document"].extend(df["Shortened Document"])
        final["Summary"].extend(df["Summary"])
        final["Document length"].extend(df["Document length"])
        final["Shortened Document length"].extend(df["Shortened Document length"])
    if mode == "train_set":
        print(len(final["Sample ids"]))
        assert len(final["Sample ids"]) == 4795 # 4795 2396 2277 181
    elif mode == "val_set":
        assert len(final["Sample ids"]) == 273 # 273 120 133 11 
    else:
        assert len(final["Sample ids"]) == 266 # 266 129 129 11
#     print(f"""{mode} LEN: {len(final["Sample ids"])}""")
    final_df = pd.DataFrame(final)
    final_df.to_csv(os.path.join(path, f"""{mode}.csv"""))
    print("SAVE TO CSV FINISHED")

parser = argparse.ArgumentParser()
parser.add_argument("--approach", help="specify extraction approach", type=str)
parser.add_argument("--shuffle", default=False, help="select unshuffled or shuffled", type=bool) 
parser.add_argument("--seed", help="specify seed of shuffled dataset", type=int) 
args = parser.parse_args()

if __name__ == "__main__":
    combine(args, mode = "train_set")
    combine(args, mode = "val_set")
    combine(args, mode = "test_set")
    