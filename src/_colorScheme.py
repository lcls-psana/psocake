import os
# Color scheme
if 'LCLS' in os.environ['PSOCAKE_FACILITY'].upper():
    plotForeground = cardinalRed_hex = str("#8C1515") # Cardinal red
    plotBackground = sandstone = '#D2C295' # Sandstone
    selectedFG = sandstone # Sandstone: selected background
    selectedBG = cardinalRed = '#8C1515' # Cardinal red: selected foreground
    selectedBorder = black = '#2E2D29' # selected border
    unselectedFG = coolGray = '#4D4F53' # Cool gray: unselected foreground
    unselectedBG = sandstone # unselected background
    unselectedBorder = beige = '#D2C295' # Beige: unselected border
    foreground = cardinalRed # dark red
    background = grayLite = str("#ECECEC") # light grey
    pixelInfo = cardinalRed
    maskInfo = black
    mouseBackground = sandstone
elif 'PAL' in os.environ['PSOCAKE_FACILITY'].upper():
    plotForeground = wine = '#722F37'  # wine: selected foreground
    plotBackground = gray_hex = str("#3f3c30") # gray
    selectedFG = sandstone = '#D2C295'  # Sandstone: selected background
    selectedBG = wine  # wine: selected foreground
    selectedBorder = black = '#2E2D29'  # selected border
    unselectedFG = coolGray = '#4D4F53'  # Cool gray: unselected foreground
    unselectedBG = sandstone  # unselected background
    unselectedBorder = beige = '#D2C295'  # Beige: unselected border
    foreground = darkRed_hex = str("#820000") # dark red
    background = beige_hex = ("#9d9573") # beige
    pixelInfo = wine
    maskInfo = black80_hex = str("#585754") # black 80%
    mouseBackground = beige_hex


# Extra
#    black80_hex = str("#585754") # black 80%
#    gray_hex = str("#3f3c30") # gray
#    gray90_hex = str("#565347") # gray 90%
#    gray60_hex = str("#8a887d") # gray 60%
#    beige_hex = ("#9d9573") # beige

