import os
# Color scheme
if 'LCLS' in os.environ['PSOCAKE_FACILITY'].upper():
    plotForeground = cardinalRed_hex = str("#8C1515") # Cardinal red
    plotBackground = sandstone100_rgb = (221,207,153) # Sandstone
    foreground = darkRed_hex = str("#820000") # dark red
    background = grayLite_hex = str("#ececec") # light grey
    pixelInfo = cardinalRed_hex
    maskInfo = black_hex = str("#2e2d29") # black
    mouseBackground = sandstone100_rgb
elif 'PAL' in os.environ['PSOCAKE_FACILITY'].upper():
    plotForeground = cardinalRed_hex = str("#8C1515") # Cardinal red
    plotBackground = gray_hex = str("#3f3c30") # gray
    foreground = darkRed_hex = str("#820000") # dark red
    background = beige_hex = ("#9d9573") # beige
    pixelInfo = cardinalRed_hex
    maskInfo = black80_hex = str("#585754") # black 80%
    mouseBackground = beige_hex

#    black80_hex = str("#585754") # black 80%
#    gray_hex = str("#3f3c30") # gray
#    gray90_hex = str("#565347") # gray 90%
#    gray60_hex = str("#8a887d") # gray 60%
#    beige_hex = ("#9d9573") # beige

