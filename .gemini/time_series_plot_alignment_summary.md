# Time-Series Plot Alignment Summary

## Changes Made

### Problem
The Time-Series Tab had inconsistent plot styles across three analysis levels:
1. **Population Level**: Clean plot with caption text embedded in the figure
2. **Trial Level**: Different design with background colors, DBS badges, and event markers
3. **Session Level**: Clean plot with caption text embedded in the figure

Additionally, caption text was appearing as part of the plot figure itself (using plot annotations), which made it part of the exported image rather than dashboard UI.

### Solution

#### 1. Unified Plot Styling
All three levels (Population, Trial, Session) now use the **same base plot design**:
- Clean, minimal appearance
- Consistent use of `create_base_time_series_figure()`
- DBS ON/OFF differentiated by line color and legend only
- No background colors, badges, or special decorations
- Consistent line widths using `PLOT_STYLE.line_width_normal`

**Files Modified:**
- `dashboard/time_series_plots.py`
  - `plot_channel_time_series()` - Simplified to match unified style
  - `plot_multi_channel_time_series()` - Simplified to match unified style
  - `plot_population_time_series()` - Removed embedded caption
  - `plot_session_comparison_time_series()` - Removed embedded caption

#### 2. Caption Text Separation
Captions are now **dashboard elements** rather than plot annotations:
- Removed `add_caption_below()` calls from all plotting functions
- Added `st.caption()` calls in `time_series_tab.py` after each plot
- Captions appear below plots but are not part of the exported figure

**Files Modified:**
- `dashboard/time_series_tab.py`
  - Added captions for neural channel plots (single and multi-channel)
  - Added captions for population level plots
  - Added captions for session level plots

#### 3. Code Cleanup
- Removed unused `add_caption_below` import from `time_series_plots.py`
- Removed custom background colors, event markers, and DBS badges from trial-level plots

## Result

All plots in the Time-Series Tab now have:
✅ **Consistent visual style** across all three levels
✅ **Clean, professional appearance** without distracting decorations
✅ **Proper separation** between plot data and descriptive text
✅ **Informative captions** as dashboard UI elements (not part of plot)

## Color Scheme
- **DBS ON**: Red (`#ff0035` - strawberry_red)
- **DBS OFF**: Purple (`#59546c` - vintage_grape), dashed line
- Participant-specific colors maintained for population-level comparisons
