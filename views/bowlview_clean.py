# This file will be used to replace the corrupted bowlview.py
# The content is just the core ending we need

        ###-------------------------------------RECORDS STATS-------------------------------------###
        # Records Stats Tab
        with tabs[11]:
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Bowling Records</h3>", unsafe_allow_html=True)
            st.write("Coming soon...")

    else:
        st.warning("Please upload bowling statistics to view this page.")

# Call the function
if __name__ == "__main__":
    display_bowl_view()
