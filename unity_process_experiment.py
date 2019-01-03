from unityagents.environment import UnityEnvironment
from deeprl.environment.resources import banana



def create_environment():
    env = UnityEnvironment(banana, 
                           docker_training=True, 
                           no_graphics=True)
    




if __name__ == '__main__':
    from multiprocessing import Process
    p = Process(target=create_environment)
    p.start()
    p.join()