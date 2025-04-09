import csv
import json
import os

INPUT_FOLDER = "./../data/raw_data/2025_raw"
OUTPUT_CSV = "ipl_squad_data.csv"

CSV_FIELDS = [
    "Player Name",
    "Delivery Name",
    "Role",
    "Batting Style",
    "Bowling Style",
    "Team",
    "Is Overseas",
]

TEAM_NAME_MAPPING = {
    "csk": "Chennai Super Kings",
    "dc": "Delhi Capitals",
    "kkr": "Kolkata Knight Riders",
    "mi": "Mumbai Indians",
    "pk": "Punjab Kings",
    "rr": "Rajasthan Royals",
    "rcb": "Royal Challengers Bangalore",
    "srh": "Sunrisers Hyderabad",
    "gt": "Gujarat Titans",
    "lsg": "Lucknow Super Giants",
}


def extract_player_data(player, team_name, is_overseas):
    return {
        "Player Name": player.get("longName", "Unknown"),
        "Delivery Name": player.get("name", "Unknown"),
        "Role": ", ".join(player.get("playingRoles", [])) or "Unknown",
        "Batting Style": ", ".join(player.get("longBattingStyles", [])) or "Unknown",
        "Bowling Style": ", ".join(player.get("longBowlingStyles", [])) or "Unknown",
        "Team": team_name,
        "Is Overseas": "Yes" if is_overseas else "No",
    }


def process_json_files():

    for file_name in os.listdir(INPUT_FOLDER):
        all_players = []
        if file_name.endswith(".json"):

            team_name = TEAM_NAME_MAPPING[file_name.replace("_raw.json", "")]
            with open(
                os.path.join(INPUT_FOLDER, file_name), "r", encoding="utf-8"
            ) as f:
                data = json.load(f)
                players = data.get("players", [])
                for player_data in players:
                    all_players.append(
                        extract_player_data(
                            player_data["player"],
                            team_name,
                            player_data.get("isOverseas", False),
                        )
                    )

        with open(
            team_name.replace(" ", "_") + "_squad.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(all_players)

        print(f"CSV file created successfully: {team_name.replace(' ', '')}_squad")


if __name__ == "__main__":
    process_json_files()
