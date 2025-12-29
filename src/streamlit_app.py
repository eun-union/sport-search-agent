"""
Sports Search Agent - Streamlit Frontend

A professional UI for the sports search agent powered by Flyte v2.
"""

import asyncio
import base64
import time
import msgpack
import streamlit as st
import flyte

# Page configuration
st.set_page_config(
    page_title="Sports Search Agent",
    page_icon="🏆",
    layout="centered",
)

# Custom CSS for professional look
st.markdown("""
<style>
    .stTextInput > div > div > input {
        font-size: 18px;
    }
    .source-link {
        color: #1E88E5;
        text-decoration: none;
    }
    .source-link:hover {
        text-decoration: underline;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


def init_flyte():
    """Initialize Flyte connection."""
    if "flyte_initialized" not in st.session_state:
        flyte.init_from_config()
        st.session_state.flyte_initialized = True


def run_search(query: str):
    """Execute the sports search workflow."""
    from src.sports_search_agent import sports_search_agent

    run = flyte.run(sports_search_agent, query=query, max_results=5)
    return run


def get_result(run):
    """Get the result from a completed run."""
    outputs = run.outputs()
    output_dict = outputs.to_dict()
    binary_value = output_dict['literals'][0]['value']['scalar']['binary']['value']
    decoded = msgpack.unpackb(base64.b64decode(binary_value), raw=False)
    return decoded


def poll_run_status(run_name: str):
    """Poll run status until completion."""
    import flyte.remote  # Lazy import to avoid event loop issues
    from flyte.remote._action import ActionPhase

    run = flyte.remote.Run.get(name=run_name)

    terminal_phases = {
        ActionPhase.SUCCEEDED,
        ActionPhase.FAILED,
        ActionPhase.ABORTED,
        ActionPhase.TIMED_OUT,
    }

    while True:
        phase = run.phase
        if phase in terminal_phases:
            return run, phase
        time.sleep(2)
        run = flyte.remote.Run.get(name=run_name)  # Refresh


# Main app
def main():
    # Header
    st.title("🏆 Sports Search Agent")
    st.markdown("*Powered by Flyte v2, Tavily, and Anthropic Claude*")
    st.divider()

    # Initialize Flyte
    init_flyte()

    # Search input
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "Ask a sports question:",
            placeholder="Who won the latest Super Bowl?",
            label_visibility="collapsed",
        )
    with col2:
        search_button = st.button("Search", type="primary", use_container_width=True)

    # Example queries
    st.markdown("**Try these:**")
    example_cols = st.columns(3)
    examples = [
        "Who won the 2024 NBA Finals?",
        "Latest FIFA World Cup winner",
        "Current Formula 1 standings",
    ]

    for col, example in zip(example_cols, examples):
        if col.button(example, use_container_width=True):
            query = example
            search_button = True

    st.divider()

    # Execute search
    if search_button and query:
        st.markdown(f"### Searching: *{query}*")

        # Start the workflow
        with st.status("Running sports search agent...", expanded=True) as status:
            st.write("🚀 Starting Flyte workflow...")
            run = run_search(query)

            st.write(f"📋 Run ID: `{run.name}`")
            st.write(f"🔗 [View in Union.ai Console]({run.url})")

            st.write("🔍 Searching the web with Tavily...")
            st.write("🤖 Synthesizing response with Claude...")

            # Poll for completion
            try:
                from flyte.remote._action import ActionPhase
                run_obj, final_phase = poll_run_status(run.name)

                if final_phase == ActionPhase.SUCCEEDED:
                    status.update(label="Search complete!", state="complete")
                    # Store result in session state
                    st.session_state.result = get_result(run_obj)
                    st.session_state.search_success = True
                else:
                    status.update(label=f"Failed: {final_phase}", state="error")
                    st.session_state.search_success = False
                    st.session_state.error_msg = f"Workflow failed with status: {final_phase}"

            except Exception as e:
                status.update(label="Error occurred", state="error")
                st.session_state.search_success = False
                import traceback
                st.session_state.error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"

        # Display results outside the status block
        if st.session_state.get("search_success"):
            result = st.session_state.result

            # Display answer
            st.markdown("### Answer")
            st.info(result['answer'])

            # Display sources
            if result['sources']:
                st.markdown("### Sources")
                for i, url in enumerate(result['sources'], 1):
                    st.markdown(f"{i}. [{url}]({url})")

            # Stats
            st.caption(f"Used {result['search_results_count']} search results")

        elif st.session_state.get("search_success") is False:
            st.error(st.session_state.get("error_msg", "Unknown error"))

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #888;">
        <small>
            Built with Flyte v2 |
            <a href="https://union.ai" target="_blank">Union.ai</a> |
            Multi-step AI orchestration demo
        </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
