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

#############################

# F Y y_old y_new Y_in Y_out
# initial_conditions ode_inputs ode_states ode_outputs
# ode_ICs params states outputs

def get_name(var_type, state_name, i_step=None, i_stage=None, j_stage=None):
    name = '{}:{}'.format(var_type, state_name)

    if j_stage is not None:
        name = 'stage{}_{}'.format(j_stage, name)

    if i_stage is not None:
        name = 'stage{}_{}'.format(i_stage, name)

    if i_step is not None:
        name = 'step{}_{}'.format(i_step, name)

    return name
