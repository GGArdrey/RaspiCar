class State:
    def __init__(self, fsm):
        self.fsm = fsm

    def enter(self, *args):
        pass

    def exit(self, *args):
        pass

    def throttle(self, *args):
        pass

    def steer(self, *args):
        pass

    def sensors_enable(self, *args):
        # Note: This is a placeholder for the actual implementation
        # As TOF sensors are not used in the current implementation
        # Nevertheless, this is the default behavior that they can be enabled/disabled in every state
        self.fsm.car.start_sensors()

    def sensors_disable(self, *args):
        # Note: This is a placeholder for the actual implementation
        # As TOF sensors are not used in the current implementation
        # Nevertheless, this is the default behavior that they can be enabled/disabled in every state
        self.fsm.car.stop_sensors()

    def manual_emergency(self, *args):
        pass

    def reset_manual_emergency(self, *args):
        pass

    def ping_timeout(self, *args):
        pass

    def ping_success(self, *args):
        pass

    def front_tof_measurement(self, *args):
        pass

    def rear_tof_measurement(self, *args):
        pass

class IdleState(State):
    def enter(self, *args):
        self.fsm.car.motor_drive(0)
        self.fsm.car.servo_steer(0)
        self.fsm.car.flash_onboard_led(1)

    def exit(self, *args):
        self.fsm.car.stop_onboard_led()

    def throttle(self, *args):
        self.fsm.car.motor_drive(int(args[0]))
        self.fsm.transition_to('DRIVING')

    def steer(self, *args):
        self.fsm.car.servo_steer(int(args[0]))
        self.fsm.transition_to('DRIVING')

    def manual_emergency(self, *args):
        self.fsm.transition_to('MANUAL_EMERGENCY_STOP')

    def ping_timeout(self, *args):
        self.fsm.transition_to('AUTOMATIC_EMERGENCY_STOP')


class DrivingState(State):

    def enter(self, *args):
        self.fsm.car.flash_onboard_led(3)

    def exit(self, *args):
        self.fsm.car.stop_onboard_led()

    def throttle(self, *args):
        self.fsm.car.motor_drive(int(args[0]))

    def steer(self, *args):
        self.fsm.car.servo_steer(int(args[0]))

    def manual_emergency(self, *args):
        self.fsm.transition_to('MANUAL_EMERGENCY_STOP')

    def ping_timeout(self, *args):
        self.fsm.transition_to('AUTOMATIC_EMERGENCY_STOP')


class ManualEmergencyStopState(State):

    def enter(self, *args):
        self.fsm.car.motor_drive(0)
        self.fsm.car.flash_onboard_led(10)

    def exit(self, *args):
        self.fsm.car.stop_onboard_led()

    def reset_manual_emergency(self, *args):
        self.fsm.transition_to('IDLE')

class AutomaticEmergencyStopState(State):
    def enter(self, *args):
        self.fsm.car.motor_drive(0)

        self.fsm.car.flash_onboard_led(10)

    def exit(self, *args):
        self.fsm.car.stop_onboard_led()
    def ping_success(self, *args):
        self.fsm.transition_to('IDLE')

    def manual_emergency(self, *args):
        self.fsm.transition_to('MANUAL_EMERGENCY_STOP')

class DistanceEmergencyStopState(State):
    def enter(self, *args):
        self.fsm.car.motor_drive(0)

        self.fsm.car.flash_onboard_led(10)

    def exit(self, *args):
        self.fsm.car.stop_onboard_led()

    def throttle(self, *args):
        if int(args[0]) < 0:
            self.fsm.car.motor_drive(int(args[0]))

    def steer(self, *args):
        self.fsm.car.servo_steer(int(args[0]))

    def emergency_enable(self, *args):
        self.fsm.transition_to('MANUAL_EMERGENCY_STOP')

    def ping_timeout(self, *args):
        self.fsm.transition_to('AUTOMATIC_EMERGENCY_STOP')


class FiniteStateMachine:
    def __init__(self, car):
        self.car = car
        self.states = {
            'IDLE': IdleState(self),
            'DRIVING': DrivingState(self),
            'MANUAL_EMERGENCY_STOP': ManualEmergencyStopState(self),
            'AUTOMATIC_EMERGENCY_STOP': AutomaticEmergencyStopState(self)
            #'DISTANCE_EMERGENCY_STOP': DistanceEmergencyStopState(self)
        }
        self.state = self.states['IDLE']
        self.state.enter()

    def transition_to(self, state_name):
        '''
        Transitions the FSM to a different state
        '''
        print(f"Transitioning to {state_name}")
        self.state.exit()
        self.state = self.states[state_name]
        self.state.enter()

    def handle_event(self, event, *args):
        '''
        Input alphabet to the FSM are always handled with this method.
        '''
        try:
            method = getattr(self.state, event)
            if method:
                method(*args)
        except Exception as e:
            print(f"FSM Error handling event \"{event}\" in state \"{self.state.__class__.__name__}\": ", e)
