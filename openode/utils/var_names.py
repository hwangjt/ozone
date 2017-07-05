def get_F_name(j_stage, state_name):
    return 'F_%i:%s' % (j_stage, state_name)

def get_y_old_name(state_name):
    return 'y_old:%s' % state_name

def get_y_new_name(state_name):
    return 'y_new:%s' % state_name

def get_Y_name(i_stage, state_name):
    return 'Y_%i:%s' % (i_stage, state_name)

def get_step_name(i_step, state_name):
    return 'step_%i:%s' % (i_step, state_name)
