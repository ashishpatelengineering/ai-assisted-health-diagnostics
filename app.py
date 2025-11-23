import os
import time
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai


def main():
    # Page configuration
    st.set_page_config(
        page_title="AI Assisted Health Diagnostics",
    )

    st.title("AI Assisted Health Diagnostics")
    st.write(
        """
        This app reviews a patient's video in which they describe their symptoms or
        medical concerns. It then generates a clear and structured summary to support
        healthcare professionals in understanding the patient's condition more
        quickly and making informed decisions.
        """
    )

    # Input API key through Streamlit UI
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    # Configure the API key
    genai.configure(api_key=api_key)

    @st.cache_resource
    def initialize_agent():
        return Agent(
            name="Video AI Summarizer",
            model=Gemini(id="gemini-2.0-flash"),
            tools=[DuckDuckGo()],
            markdown=True,
        )

    # Initialize the agent
    try:
        multimodal_agent = initialize_agent()
    except Exception:
        st.error(
            "Failed to initialize the AI agent. Please check your API Key."
        )
        st.stop()

    # File uploader
    video_file = st.sidebar.file_uploader(
        "Upload a Video File:",
        type=["mp4", "mov", "avi"],
        help="Upload a Video File",
    )

    if video_file:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ) as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name

        st.video(video_path, format="video/mp4", start_time=0)

        user_query = st.text_area(
            "What information would you like to extract from the patient's video?",
            placeholder=(
                "Examples: Summarize the patient's main symptoms and concerns, "
                "Identify any urgent medical issues mentioned, Extract details "
                "about symptom duration and severity, Note the patient's emotional "
                "state and pain description, List any medications or treatments "
                "mentioned"
            ),
        )

        if st.button("Analyse Video", key="analyze_video_button"):
            if not user_query:
                st.warning(
                    "Please enter a question or insight to analyze the video."
                )
            else:
                try:
                    with st.spinner(
                        "Processing video and gathering insights..."
                    ):
                        processed_video = upload_file(video_path)

                        while processed_video.state.name == "PROCESSING":
                            time.sleep(1)
                            processed_video = get_file(
                                processed_video.name
                            )

                        analysis_prompt = f"""
You are an AI medical intake assistant analyzing a patient's video description
of their symptoms.

CRITICAL MEDICAL DISCLAIMERS:
- You are an assistant for information organization ONLY
- You MUST NOT provide any medical diagnoses
- You MUST NOT suggest specific treatments or medications
- You MUST flag any potentially urgent symptoms for professional review
- All information should be verified by qualified healthcare providers

ANALYSIS FRAMEWORK:
Please analyze this patient video and provide a structured clinical summary
focusing on:

1. Chief Complaint and Presenting Symptoms
   - Primary reason for seeking care
   - Main symptoms described in patient's own words

2. Symptom Characterization
   - Duration and timing patterns
   - Severity using patient's own descriptors
   - Precipitating and alleviating factors
   - Associated symptoms

3. Relevant Medical Context
   - Any mentioned medications, allergies, or existing conditions
   - Previous treatments or consultations mentioned
   - Family history if disclosed

4. Patient Perspective and Emotional State
   - Patient's main concerns and worries
   - Emotional tone and affect observed
   - Impact on daily activities and quality of life

5. Clinical Documentation Support
   - Key phrases and direct patient quotations
   - Inconsistencies or gaps in the history
   - Potential areas for further clarification

SPECIFIC USER REQUEST:
{user_query}

Please structure your response using clear headings and bullet points for easy
clinical review.
"""

                        response = multimodal_agent.run(
                            analysis_prompt, videos=[processed_video]
                        )

                    st.subheader("Analysis Result")
                    st.markdown(response.content)

                except Exception as error:
                    error_message = str(error)

                    if "API_KEY_INVALID" in error_message:
                        st.error(
                            "The provided API key is invalid. Please check and "
                            "enter a valid key."
                        )
                    else:
                        st.error(
                            "An unexpected error occurred during analysis. "
                            "Please try again later."
                        )
                        st.write(error)
                finally:
                    Path(video_path).unlink(missing_ok=True)
    else:
        st.info("Please upload a video file to proceed.")


if __name__ == "__main__":
    main()
