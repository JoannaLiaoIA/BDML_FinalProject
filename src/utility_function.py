import re
import io
import requests
import numpy as np
import pandas as pd

class Sample:
    model: pd.DataFrame
    model_label: str
    algorithm: str
    parameters: dict
    sample_label: str
    filename_label: str

    def __init__(self, model: pd.DataFrame, model_label: str, algorithm: str, parameters: dict):
        self.model = model
        self.model_label = model_label
        self.algorithm = algorithm
        self.parameters = parameters

        param_str = ", ".join(f"{k} = {v}" for k, v in self.parameters.items())
        self.sample_label = f"{self.algorithm} {self.model_label} ({param_str} - {self.model['Year'].min()}Q{self.model['Quarter'].min()} to {self.model['Year'].max()}Q{self.model['Quarter'].max()})"

        param_str = "_".join(f"{k.replace("_", "")}={v}" for k, v in self.parameters.items())
        self.filename_label = f"{self.algorithm}_{self.model_label.replace(" ", "_")}_{param_str}_{self.model['Year'].min()}Q{self.model['Quarter'].min()}_{self.model['Year'].max()}Q{self.model['Quarter'].max()}"

    def print_parameters(self) -> None:
        """
        Print the parameters in a readable format.
        """
        title = f"+---------- {self.model_label} Parameters ----------+"

        print(title)
        for key, value in self.parameters.items():
            print(f"| {key}: {value}", " " * (len(title) - len(f"| {key}: {value}") - 2) + "|", sep = "")
        print("+" + "-" * (len(title) - 2) + "+", sep = "")

def set_training_periods(training_start_date: str, training_end_date: str) -> None:
    """
    Set the global variables for training periods based on the provided start and end dates.
    """
    TRAINING_START_YEAR = int(training_start_date.split("-")[0])
    TRAINING_END_YEAR = int(training_end_date.split("-")[0])

    TRAINING_START_QUARTER = (int(training_start_date.split("-")[1]) - 1) // 3 + 1
    TRAINING_END_QUARTER = (int(training_end_date.split("-")[1]) - 1) // 3 + 1
    
    return TRAINING_START_YEAR, TRAINING_END_YEAR, TRAINING_START_QUARTER, TRAINING_END_QUARTER


def set_holdout_periods(holdout_start_date: str, holdout_end_date: str) -> None:
    """
    Set the global variables for holdout periods based on the provided start and end dates.
    """
    HOLDOUT_START_YEAR = int(holdout_start_date.split("-")[0])
    HOLDOUT_END_YEAR = int(holdout_end_date.split("-")[0])

    HOLDOUT_START_QUARTER = (int(holdout_start_date.split("-")[1]) - 1) // 3 + 1
    HOLDOUT_END_QUARTER = (int(holdout_end_date.split("-")[1]) - 1) // 3 + 1

    return HOLDOUT_START_YEAR, HOLDOUT_END_YEAR, HOLDOUT_START_QUARTER, HOLDOUT_END_QUARTER


def describe_data(df: pd.DataFrame, title: str, n: int = 3) -> None:
    """
    Print the head and tail n rows, then check "infinite", "NA" and the datatype of each columns.
    """

    title_str = f"| This is {title} |"
    num = len(title_str)

    print("\n")
    print("+", "-" * (num - 2), "+", sep = "")
    print(title_str)
    print("+", "-" * (num - 2), "+", sep = "")

    print("\n", f"| Dataset Shape: {df.shape}", "\n", sep = "")

    print(f"========== first {n} rows ==========")
    print(df.head(n), "\n")

    print(f"========== last {n} rows ==========")
    print(df.tail(n), "\n")

    print(f"========== infinite check ==========")
    print(df.isin([np.inf, -np.inf]).sum(), "\n")

    print(f"========== NA check ==========")
    print(df.isna().sum(), "\n")

    print(f"========== info of data ==========")
    df.info()
    print("\n")


def convert_date(date_str: str):
    if pd.isna(date_str) or str(date_str).strip() == "":
        return pd.NaT
    
    def check_year(y: int) -> int:
        return y + 1911 if y < 200 else y

    s = str(date_str).strip()
    s = s.split(" ")[0]

    # x 年 x 月
    if "年" in s and "月" in s:
        match = re.search(r"(\d+)\s*年\s*(\d+)\s*月", s)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            year = check_year(year)
            return f"{year:04d}-{month:02d}-00"
        
    # x 年
    elif "年" in s:
        match = re.search(r"(\d+)\s*年", s)
        if match:
            year = int(match.group(1))
            year = check_year(year)
            return f"{year:04d}-00-00"
        
    # yymmdd or yyymmdd or yyyymmdd e.g. 920912, 1110912, 20210912
    elif s.isdigit() and (len(s) >= 6 and len(s) <= 7):
        year = int(s[:-4])
        month = int(s[-4:-2])
        day = int(s[-2:])
        year = check_year(year)
        return f"{year:04d}-{month:02d}-{day:02d}"

    # yy or yyy or yyyy
    elif s.isdigit() and (len(s) >= 2 and len(s) <= 4):
        year = int(s)
        year = check_year(year)
        return f"{year:04d}-00-00"
    
    # yyyMmm or yyyyMmm e.g. 111M09, 2021M09
    elif "M" in s:
        year = int(s[:-3])
        month = int(s[-2:])
        year = check_year(year)
        return f"{year:04d}-{month:02d}-00"
    
    # yyy/mm/dd or yyyy/mm/dd or yyy/mm or yyyy/mm e.g. 111/09/12, 2021/09/12
    # yyy-mm-dd or yyyy-mm-dd or yyy-mm or yyyy-mm e.g. 111-09-12, 2021-09-12
    elif "/" in s or "-" in s:
        s = s.replace("/", "-")
        parts = s.split("-")

        year = int(parts[0])
        month = int(parts[1]) if len(parts) >= 2 else 0
        day = int(parts[2]) if len(parts) >= 3 else 0
        year = check_year(year)
        return f"{year:04d}-{month:02d}-{day:02d}"
    
    return "Invalid Date Format"


def get_file_from_drive(file_url: str) -> pd.DataFrame:
    """
    Download a CSV file from Google Drive or Google Sheets and return it as a DataFrame.
    """
    if not file_url:
        raise ValueError("File url cannot be empty")

    target_url = file_url
    params = None
    is_drive_file = False

    if "docs.google.com/spreadsheets" in file_url:
        print("Detected Google Sheets. Converting to CSV export URL...")
        if any(keyword in file_url for keyword in ["/edit", "/view", "htmlview"]):
            target_url = re.sub(r"/(edit|view|htmlview).*$", "/export?format=csv", file_url)
        elif "export" in file_url and "format=csv" not in file_url:
            target_url = re.sub(r"format=[^&]+", "format=csv", file_url)
            if "format=csv" not in target_url:
                target_url += "&format=csv"

    elif "drive.google.com/file" in file_url:
        print("Detected Google Drive CSV file. Preparing to fetch...")
        file_id_match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", file_url)
        if file_id_match:
            file_id = file_id_match.group(1)
            target_url = "https://drive.google.com/uc"
            params = {"id": file_id, "export": "download", "confirm": "true"}
            is_drive_file = True
        else:
            raise ValueError("Failed to parse Google Drive file ID")

    else:
        print("Detected standard CSV URL...")

    print(f"Fetching data from {target_url}...")
    
    try:
        # 使用 Session 來保持連線與 Cookies (這對繞過警告很重要)
        session = requests.Session()
        response = session.get(target_url, params = params, stream = True, timeout = 30)
        
        # 檢查是否遇到「檔案過大無法掃描病毒」的警告
        if is_drive_file:
            for key, value in response.cookies.items():
                if key.startswith("download_warning"):
                    print("Large file detected. Bypassing virus scan warning...")
                    # 將抓到的 token 塞進參數裡再次請求
                    params["confirm"] = value
                    response = session.get(target_url, params = params, stream = True, timeout = 30)
                    break

        response.raise_for_status()
        
        # 再次確認最終抓下來的是不是 HTML
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type:
            raise PermissionError("Downloaded content is STILL HTML.")

        # 讀取大檔案可能會花一點時間
        df = pd.read_csv(io.BytesIO(response.content), encoding = "utf-8-sig")
        print("Download finished!")
        return df
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error occurred while fetching the file: {e}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse CSV. The file format might be incorrect: {e}")
    

