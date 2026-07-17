import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

def analyze_combined_telemetry():
    print("[INFO] Initializing Analytics Engine...")
    
    usage_file = os.path.expanduser("~/laptop_usage.csv")
    focus_file = "focus_ergonomics_log.csv"
    
    usage_data_str = "No laptop usage data available."
    if os.path.exists(usage_file):
        try:
            df_usage = pd.read_csv(usage_file)
            usage_data_str = df_usage.to_string(index=False)
        except Exception as e:
            print(f"[ERROR] Failed to read system telemetry: {e}")

    print("[INFO] Processing local streaming metrics...")
    focus_data_str = "No focus data available."
    if os.path.exists(focus_file):
        try:
            df_focus = pd.read_csv(focus_file)
            df_focus['Timestamp'] = pd.to_datetime(df_focus['Timestamp'])
            df_focus['Is_Distracted'] = (df_focus['Focus_State'] == 'Distracted').astype(int)
            df_focus['Hour'] = df_focus['Timestamp'].dt.hour
            
            hourly_summary = df_focus.groupby('Hour').agg(
                Total_SecondsLogged=('Focus_State', 'count'),
                Distracted_Seconds=('Is_Distracted', 'sum')
            ).reset_index()
            
            hourly_summary['Distraction_Percent'] = (hourly_summary['Distracted_Seconds'] / hourly_summary['Total_SecondsLogged']) * 100
            hourly_summary['Distraction_Percent'] = hourly_summary['Distraction_Percent'].round(1)
            
            focus_data_str = hourly_summary.to_string(index=False)
        except Exception as e:
            print(f"[ERROR] Pandas pipeline transformation breakdown: {e}")

    if usage_data_str == "No laptop usage data available." and focus_data_str == "No focus data available.":
        print("[CRITICAL] Pipeline empty. Execution halted.")
        return

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )
    
    system_instruction = """
    You are an advanced productivity and ergonomics analyst. You are receiving two sets of telemetry data:
    1. Laptop Usage Logs: Raw system wake/sleep events.
    2. Focus & Ergonomics Logs: Hourly aggregated webcam data showing total seconds tracked and time spent 'Distracted'.
    
    Correlate these datasets. Identify specific hours where the user is most distracted, calculate the worst distraction percentage, and provide concise, actionable insights on how they might restructure their work blocks.
    Format cleanly using Markdown headers and bullet points. Do not output raw code.
    """
    
    prompt = f"--- LAPTOP USAGE DATA ---\n{usage_data_str}\n\n--- HOURLY FOCUS SUMMARY ---\n{focus_data_str}"
    
    print("\n[INFO] Injecting telemetry matrix into Gemini 2.5 Flash Engine...")
    
    try:
        messages = [SystemMessage(content=system_instruction), HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        
        print("\n========================================================")
        print("          PRODUCTIVITY & ERGONOMIC ANALYSIS             ")
        print("========================================================\n")
        print(response.content)
        
    except Exception as e:
        print(f"\n[CRITICAL] LLM Network Framework Failure: {e}")

if __name__ == "__main__":
    analyze_combined_telemetry()
