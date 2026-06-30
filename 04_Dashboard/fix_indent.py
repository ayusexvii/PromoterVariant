with open('app.py', 'r') as f:
    lines = f.readlines()

# Find the line with "tissue_clean" to locate the block
start_idx = None
for i, line in enumerate(lines):
    if 'tissue_clean = tissue.split' in line:
        start_idx = i
        break

if start_idx is not None:
    # We want to rewrite from start_idx to the next radio block end
    # We'll set the indentation for the whole block to 4 spaces for the main block
    # and 8 spaces for the nested lines.
    # Let's just replace the entire section from tissue_clean to the closing parenthesis of radio.
    # We'll manually rebuild that section.
    
    # Find the end of the radio block (the line with ')')
    end_idx = None
    for i in range(start_idx, len(lines)):
        if ')' in lines[i] and 'st.sidebar.radio' not in lines[i]:
            # This is likely the closing parenthesis
            end_idx = i
            break
    
    if end_idx is not None:
        # Build new lines
        new_lines = [
            '    # --- Tissue Selector ---\n',
            '    st.sidebar.markdown("### 🧬 Tissue")\n',
            '    tissue_options = ["Liver", "Whole Blood (v2.6)", "Brain (v2.6)"]\n',
            '    tissue = st.sidebar.selectbox("Select Tissue", tissue_options)\n',
            '\n',
            '    # Normalize tissue name\n',
            '    tissue_clean = tissue.split(" (")[0] if "(" in tissue else tissue\n',
            '    st.sidebar.caption(f"Current: {tissue_clean}")\n',
            '\n',
            '    page = st.sidebar.radio(\n',
            '        "NAVIGATION",\n',
            '        ["1. Dashboard Overview", "2. Gene Locus Explorer", "3. Live Mutation Simulator", "4. Deep Model Insights"]\n',
            '    )\n'
        ]
        
        # Replace the lines
        lines[start_idx:end_idx+1] = new_lines
        
        # Write back
        with open('app.py', 'w') as f:
            f.writelines(lines)
        
        print("✅ Fixed indentation for the sidebar section")
    else:
        print("Could not find end of radio block")
else:
    print("Could not find tissue_clean line")

