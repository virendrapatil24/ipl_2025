import pandas as pd


def over_all_deliveries_to_per_match(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    grouped = df.groupby("match_id")
    for match_id, group in grouped:
        filename = "{match_id}.csv".format(match_id=match_id)
        filepath = f"{output_dir}/{filename}"
        group.to_csv(filepath, index=False)


# csv_path = "./../raw_data/deliveries.csv"
# output_dir = "./../raw_data/deliveries_per_match_data"
# over_all_deliveries_to_per_match(csv_path, output_dir)
