def checkbox_state_setter(widget, state):
    if widget:
        state = widget
        return state


def reset_memory(history, memory):
    memory.clear()
    history = []
    return history, history, memory
