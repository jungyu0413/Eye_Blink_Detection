from networks import *
from functions import *
import importlib



def load_net():
    experiment_name = 'pip_32_16_60_r18_l2_l1_10_1_nb10'
    data_name = 'WFLW'
    config_path = 'experiments.WFLW.pip_32_16_60_r18_l2_l1_10_1_nb10'
    my_config = importlib.import_module(config_path, package='PIPNet')
    Config = getattr(my_config, 'Config')
    cfg = Config()
    cfg.experiment_name = experiment_name 
    cfg.data_name = data_name
    meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface('/workspace/Eye_blink_detection/src/weights/meanface.txt', cfg.num_nb)
    
    resnet18 = models.resnet18(pretrained=cfg.pretrained)
    net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
    
    if cfg.use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('gpu')
    else:
        device = torch.device("cpu")
        print('cpu')
    
    net = net.to(device)

    # model, weight가 저장된 경로

    state_dict = torch.load('/workspace/Eye_blink_detection/src/weights/epoch59.pth', map_location=device) # weights into model
    # model에 적용하기 위해 인스턴스 메서드로 상태 반영
    net.load_state_dict(state_dict)
    
    return net, reverse_index1, reverse_index2, max_len, cfg, device