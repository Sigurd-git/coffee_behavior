from psychopy import visual, event, core, data, gui

win = visual.Window([1024,768], fullscr=False, units='pix')

#initialise some stimuli
fixation = visual.Circle(win, size = 5,
    lineColor = 'white', fillColor = 'lightGrey')

#run one trial
fixation.draw()
win.flip()

#wait for a response
event.waitKeys()

#cleanup
win.close()