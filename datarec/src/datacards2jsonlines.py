import huggingface_hub
import ipdb
import jsonlines
import requests
import pandas as pd
from tqdm import tqdm
import time
import argparse
import sys

# References:
# https://huggingface.co/docs/hub/en/datasets-cards
# https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/datasetcard_template.md
# https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/pipelines.ts


def fetch_datasets(n_datasets: int, page_size: int = 100):
    """
    Args:
        n_datasets: Maximum number of datasets to fetch
        page_size: Number of datasets to fetch per API call
        
    Returns:
        List of dataset information dictionaries
    """
    base_url = "https://huggingface.co/api/datasets"
    
    # Calculate number of pages needed
    n_pages = (n_datasets + page_size - 1) // page_size
    
    all_datasets = []
    
    for page in tqdm(range(1, n_pages + 1), desc="Fetching dataset pages"):
        params = {
            "sort": "downloads",
            "direction": "-1",  # Descending order
            "limit": page_size,
            "full": "true",
            "page": page
        }
        
        response = requests.get(base_url, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching page {page}: {response.status_code}")
            print(response.text)
            break            
        page_datasets = response.json()
        if not page_datasets:
            print(f"No more datasets found after page {page-1}")
            break
            
        all_datasets.extend(page_datasets)
        
        if len(all_datasets) >= n_datasets:
            all_datasets = all_datasets[:n_datasets]
            break
            
        time.sleep(0.5)
    
    print(f"Fetched information for {len(all_datasets)} datasets")
    return all_datasets

def fetch_dataset_card_content(dataset_id: str):
    """
    Fetch the README.md content for a dataset.
    
    Args:
        dataset_id: The ID of the dataset
        
    Returns:
        Content of the dataset card or None if not found
    """

    url = f"https://huggingface.co/datasets/{dataset_id}/raw/main/README.md"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to fetch card for {dataset_id}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching card for {dataset_id}: {e}")
        return None

def extract_dataset_info(datasets):
    """
    Args:
        datasets: List of dataset information dictionaries
        
    Returns:
        DataFrame with extracted dataset information
    """
    dataset_info = []
    
    for ds in datasets:
        info = {
            "id": ds.get("id", ""),
            "name": ds.get("id", "").split("/")[-1] if "/" in ds.get("id", "") else ds.get("id", ""),
            "downloads": ds.get("downloads", 0),
            "likes": ds.get("likes", 0),
            "tags": ", ".join(ds.get("tags", [])),
            "author": ds.get("author", ""),
            "lastModified": ds.get("lastModified", ""),
            "card_data": ds.get("cardData", {}),
        }
        dataset_info.append(info)
    
    return pd.DataFrame(dataset_info)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Extract HuggingFace dataset cards sorted by downloads")
    parser.add_argument("-n", "--num_datasets", type=int, default=10, help="Number of dataset cards to extract")
    args = parser.parse_args()
    dir_out = "/tmp/"
    readcards = True

    if readcards:
        with jsonlines.open("{}/datasetcards_subset.jsonl".format(dir_out), "a") as jsonl_write:
            with jsonlines.open("{}/datasetdetails_all.jsonl".format(dir_out), "r") as jsonl_read:
                for result in jsonl_read.iter(type=dict, skip_invalid=True):
                    if result["datasetcard"] is not None:
                        jsonl_write.write(result)          
    else:
        datasets = fetch_datasets(args.num_datasets)
        df = extract_dataset_info(datasets)
        df = df.sort_values("downloads", ascending=False).reset_index(drop=True)
        
        ids_found = []
        ii = 0
        with jsonlines.open("{}/datasetdetails_all.jsonl".format(dir_out), "a") as jsonl_write:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Saving dataset cards..."):
                outdict = dict()
                try:
                    dataset_id = row["id"]
                except:
                    continue
                if dataset_id in ids_found:
                    continue
                card_content = fetch_dataset_card_content(dataset_id)
                listtags = row.tags.split(",")
                if len(listtags) > len(row.card_data.keys()):
                    for i in listtags:
                        itms = i.split(":")
                        if len(itms) < 2:
                            continue
                        if itms[0] not in row.card_data.keys():
                            row.card_data[itms[0].strip()] = itms[1]
                        else:
                            continue
                
                outdict["id"] = row["id"]
                outdict["name"] = row["name"]
                outdict["downloads"] = row["downloads"]
                outdict["likes"] = row["likes"]
                outdict["author"] = row["author"]
                outdict["lastModified"] = row["lastModified"]
                for ikey in row.card_data.keys():
                    outdict[ikey] = row.card_data[ikey]
                if card_content is not None:
                    outdict["datasetcard"] = card_content 
                else:
                    outdict["datasetcard"] = None  
                # if "license" not in outdict.keys() or \
                #     outdict["license"] == "other" or \
                #         outdict["license"] == "unknown":
                #     continue
                # if "task_categories" not in outdict.keys() or \
                #         outdict["task_categories"] in ["", " ", None]:
                #     continue
                # if "language" in outdict.keys() and outdict["language"] != "en":
                #     continue
                jsonl_write.write(outdict)
                ii += 1
                ids_found.append(outdict["id"])   
