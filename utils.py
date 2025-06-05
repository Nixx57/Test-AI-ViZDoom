import vizdoom as vzd
import config
import itertools

def getAllActions():
    all_actions = []

    do_nothing_action = [False] * len(config.AVAILABLE_BUTTONS)
    all_actions.append(do_nothing_action)

    for button_to_activate in config.SINGLE_ACTION_BUTTONS:
        action_vector = [False] * len(config.AVAILABLE_BUTTONS)
        button_index = config.AVAILABLE_BUTTONS.index(button_to_activate)
        action_vector[button_index] = True
        if action_vector not in all_actions:
            all_actions.append(action_vector)

    for group in config.EXCLUSIVE_BUTTON_GROUPS:
        for button in group:
            action_vector = [False] * len(config.AVAILABLE_BUTTONS)
            button_index = config.AVAILABLE_BUTTONS.index(button)
            action_vector[button_index] = True
            if action_vector not in all_actions:
                all_actions.append(action_vector)

    flat_exclusive_buttons = [button for group in config.EXCLUSIVE_BUTTON_GROUPS for button in group]
    truly_combinable_buttons = [
        b for b in config.COMBINABLE_BUTTONS
        if b not in flat_exclusive_buttons and b not in config.SINGLE_ACTION_BUTTONS
    ]

    truly_combinable_indices = [config.AVAILABLE_BUTTONS.index(b) for b in truly_combinable_buttons]

    for combination_tuple in itertools.product([False, True], repeat=len(truly_combinable_buttons)):
        current_action_vector = [False] * len(config.AVAILABLE_BUTTONS)

        for i, is_pressed in enumerate(combination_tuple):
            if is_pressed:
                current_action_vector[truly_combinable_indices[i]] = True

        exclusive_group_options = []
        for group in config.EXCLUSIVE_BUTTON_GROUPS:
            group_choices = [[False] * len(config.AVAILABLE_BUTTONS)]
            for button in group:
                choice_vector = [False] * len(config.AVAILABLE_BUTTONS)
                choice_vector[config.AVAILABLE_BUTTONS.index(button)] = True
                group_choices.append(choice_vector)
            exclusive_group_options.append(group_choices)

        for exclusive_choices_tuple in itertools.product(*exclusive_group_options):
            combined_exclusive_vector = [False] * len(config.AVAILABLE_BUTTONS)
            for choice_vector in exclusive_choices_tuple:
                for i, value in enumerate(choice_vector):
                    if value:
                        combined_exclusive_vector[i] = True

            final_action_vector = list(current_action_vector)
            for i, value in enumerate(combined_exclusive_vector):
                if value:
                    final_action_vector[i] = True

            if final_action_vector not in all_actions:
                all_actions.append(final_action_vector)

    for button_to_activate in config.SINGLE_ACTION_BUTTONS:
        action_vector = [False] * len(config.AVAILABLE_BUTTONS)
        button_index = config.AVAILABLE_BUTTONS.index(button_to_activate)
        action_vector[button_index] = True
        if action_vector not in all_actions:
            all_actions.append(action_vector)

    print(f"Number of actions: {len(all_actions)}")

    return all_actions