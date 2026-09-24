import os
import urllib.request
import pandas as pd

def download_dataset():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(data_dir, exist_ok=True)
    target_path = os.path.join(data_dir, "student_data.csv")
    
    # URL 1: Kaggle Student Performance Factors Dataset (6,607 records, 20 features)
    urls = [
        "https://raw.githubusercontent.com/Shivi2599/Student_Performance_Factors_Kaggle/main/StudentPerformanceFactors.csv",
        "https://raw.githubusercontent.com/tevinp/Student-Performance-Factors-Analysis/main/StudentPerformanceFactors.csv",
        "https://raw.githubusercontent.com/arunkumar-subramanian/student-performance-prediction/master/student-mat.csv"
    ]
    
    success = False
    for url in urls:
        try:
            print(f"Attempting to download Kaggle dataset from: {url}")
            urllib.request.urlretrieve(url, target_path)
            df = pd.read_csv(target_path)
            if len(df) > 50:
                print(f"Successfully downloaded dataset with {len(df)} rows and {len(df.columns)} columns!")
                print("Columns:", list(df.columns))
                success = True
                break
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            
    if not success:
        raise RuntimeError("Could not download Kaggle dataset from any mirror.")

if __name__ == "__main__":
    download_dataset()
