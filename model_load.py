
from models.model import MultiViewFeatureFusion
import torch.nn as nn
import torch
import numpy as np
import yaml
import argparse
from pathlib import Path
import time
def minmaxscaler(data):
    mean = data.min()
    var = data.max()
    return (data - mean)/(var-mean)

def load_args_from_yaml(yaml_path):
    """
    从YAML文件加载模型配置参数

    Args:
        yaml_path: YAML配置文件路径

    Returns:
        args: 包含所有配置参数的argparse.Namespace对象
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {yaml_path}")

    # 加载YAML文件
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # 创建argparse.Namespace对象
    args = argparse.Namespace()

    # 设置设备
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.device = 'cpu'
    # 从YAML配置中加载参数
    args.backbone = config_dict.get('backbone', 'lenet5')
    args.cnn_output_size = config_dict.get('cnn_output_size', 64)
    args.hidden_size = config_dict.get('hidden_size', 64)
    args.rnn_type = config_dict.get('rnn_type', 'lstm')
    args.lstm_layers = config_dict.get('lstm_layers', 1)
    args.bidirectional = config_dict.get('bidirectional', True)
    args.fc_size = config_dict.get('fc_size', 64)
    args.num_domains = config_dict.get('num_domains', 5)
    args.fusion_mode = config_dict.get('fusion_mode', 'attention')
    args.method = config_dict.get('method', 'DScombine')
    args.is_sharedspecific = config_dict.get('is_sharedspecific', 0)
    args.bottleneck_dim = config_dict.get('bottleneck_dim', None)
    args.selected_features = config_dict.get('selected_features', ['RT', 'DT', 'RDT', 'ERT', 'ART'])
    args.is_domain_loss = config_dict.get('is_domain_loss', 0)
    args.optional_features = config_dict.get('optional_features', ['RT', 'DT', 'RDT', 'ERT', 'ART'])

    # 输入特征形状 - 需要转换为tuple格式
    input_feature_shapes = config_dict.get('input_feature_shapes', {})
    args.input_feature_shapes = {}
    for feature_name, shape_list in input_feature_shapes.items():
        args.input_feature_shapes[feature_name] = tuple(shape_list)

    # 类别映射
    args.class_mapping = config_dict.get('class_mapping', {})

    # 其他可能需要的参数
    args.batch_size = config_dict.get('batch_size', 24)
    args.epochs = config_dict.get('epochs', 32)
    args.lr = config_dict.get('lr', 0.001)

    print(f"✓ 成功加载配置文件: {yaml_path}")
    print(f"  - 骨干网络: {args.backbone}")
    print(f"  - 融合模式: {args.fusion_mode}")
    print(f"  - 方法: {args.method}")
    print(f"  - 选择特征: {args.selected_features}")
    print(f"  - 设备: {args.device}")

    return args

def initialize_model(args, num_classes):
    """
    Initialize the MultiViewFeatureFusion model based on the given architecture.

    Args:
        args: Arguments containing configurations and hyperparameters.
        num_classes: Number of classes.

    Returns:
        Initialized MultiViewFeatureFusion model.
    """

    # Get other fusion-related parameters from args or set defaults
    if args.fusion_mode == 'concatenate':
        method = getattr(args, 'method', 'concat')
    elif args.fusion_mode == 'attention':
        method = getattr(args, 'method', 'DScombine')


    bottleneck_dim = getattr(args, 'bottleneck_dim', None)

    # Create input_feature_shapes dictionary (should be provided in args)
    input_feature_shapes = args.input_feature_shapes  # Must be a dict mapping feature names to shapes

    # Initialize the MultiViewFeatureFusion model
    model = MultiViewFeatureFusion(
        backbone=args.backbone,
        cnn_output_size=args.cnn_output_size,
        hidden_size=args.hidden_size,
        rnn_type=args.rnn_type,
        lstm_layers=args.lstm_layers,
        bidirectional=args.bidirectional,
        fc_size=args.fc_size,
        num_domains=args.num_domains,
        input_feature_shapes=input_feature_shapes,
        fusion_mode=args.fusion_mode,
        method=method,
        is_sharedspecific=args.is_sharedspecific,
        bottleneck_dim=bottleneck_dim,
        selected_features=args.selected_features,
        is_domain_loss=args.is_domain_loss,
        num_classes=num_classes,
    )

    # Add a classifier layer to output logits for num_classes
    if method != 'DScombine':
        model.classifier = nn.Linear(args.fc_size, num_classes)

    return model


def model_load(args, best_model_path, device):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_info = torch.load(best_model_path, map_location=args.device)
    model = initialize_model(args, 7)
    model.load_state_dict(model_info["state_dict"])
    model.to(device)
    model.eval()
    return model

def model_load_with_yaml(yaml_path, model_path, device=None):
    """
    使用YAML配置文件加载模型

    Args:
        yaml_path: YAML配置文件路径
        model_path: 模型权重文件路径
        device: 设备，如果为None则自动选择

    Returns:
        model: 加载好的模型
        args: 配置参数
    """
    # 加载配置参数
    args = load_args_from_yaml(yaml_path)

    # 设置设备
    if device is not None:
        args.device = device

    # 加载模型
    model = model_load(args, model_path, args.device)

    print(f"✓ 模型加载完成")
    print(f"  - 模型路径: {model_path}")
    print(f"  - 设备: {args.device}")

    return model, args

def data_add_channel(ART_feature, DT_feature, ERT_feature, RT_feature, RDT_feature):

    ART_feature = np.expand_dims(ART_feature, axis=1)
    DT_feature = np.expand_dims(DT_feature, axis=0)
    ERT_feature = np.expand_dims(ERT_feature, axis=1)
    RT_feature = np.expand_dims(RT_feature, axis=0)
    RDT_feature = np.expand_dims(RDT_feature, axis=1)

    return ART_feature, DT_feature, ERT_feature, RT_feature, RDT_feature




def realtime_inference(args, model, ART_feature, DT_feature, ERT_feature, RT_feature, RDT_feature):
    """
    实时推理函数，基于validate_model_feature函数改写

    Args:
        args: 模型配置参数
        model: 已加载的模型
        ART_feature: ART特征数据 (numpy array)
        DT_feature: DT特征数据 (numpy array)
        ERT_feature: ERT特征数据 (numpy array)
        RT_feature: RT特征数据 (numpy array)
        RDT_feature: RDT特征数据 (numpy array)

    Returns:
        outputs: 模型预测输出
        confidence: 预测置信度 (如果适用)
    """
    device = args.device
    model.eval()

    with torch.no_grad():
        
        step_times = {}
        
        # 记录总开始时间
        start_total = time.time()
        
        # 1. 使用data_add_channel函数处理特征维度
        start_time = time.time()
        ART_processed, DT_processed, ERT_processed, RT_processed, RDT_processed = data_add_channel(
            ART_feature, DT_feature, ERT_feature, RT_feature, RDT_feature
        )
        step_times['1_处理特征维度'] = time.time() - start_time

        # 2. 将numpy数组转换为torch张量并添加batch维度，然后移动到设备
        start_time = time.time()
        feature_tensors = {
            'ART': torch.from_numpy(ART_processed).float().unsqueeze(0).to(device),  # 添加batch维度
            'DT': torch.from_numpy(DT_processed).float().unsqueeze(0).to(device),   # 添加batch维度
            'ERT': torch.from_numpy(ERT_processed).float().unsqueeze(0).to(device), # 添加batch维度
            'RT': torch.from_numpy(RT_processed).float().unsqueeze(0).to(device),   # 添加batch维度
            'RDT': torch.from_numpy(RDT_processed).float().unsqueeze(0).to(device)  # 添加batch维度
        }
        step_times['2_转换张量'] = time.time() - start_time

        # 3. 从feature_tensors中选择模型需要的特征
        start_time = time.time()
        selected_features_dict = {
            feature_name: feature_tensors[feature_name]
            for feature_name in model.selected_features
            if feature_name in feature_tensors
        }
        step_times['3_选择特征'] = time.time() - start_time

        # 4. 归一化特征
        start_time = time.time()
        selected_features_dict = {k: minmaxscaler(v) for k, v in selected_features_dict.items()}
        step_times['4_归一化特征'] = time.time() - start_time

        # 5. 模型前向推理
        start_time = time.time()
        if args.method == 'DScombine':
            fused_features, alphas, alpha_combined, u_a, u_tensor = model(selected_features_dict)
            fused_features = torch.cat(fused_features, dim=-1)  # 合并所有视角的特征
            weights = 1 - u_tensor
            outputs = alpha_combined - 1
            confidence = weights  # 使用权重作为置信度
        else:
            outputs, weights, fused_features = model(selected_features_dict)
            confidence = weights  # 使用权重作为置信度
        step_times['5_模型推理'] = time.time() - start_time
        
        # 记录总耗时
        step_times['总耗时'] = time.time() - start_total
        
        # 打印各步骤耗时
        print("\n--- 推理各步骤耗时 (秒) ---")
        for step, elapsed in step_times.items():
            print(f"{step}: {elapsed:.6f}s")
        
        return outputs, confidence


def realtime_inference_with_prediction(args, model, ART_feature, DT_feature, ERT_feature, RT_feature, RDT_feature, gesture_dict=None):
    """
    实时推理函数，返回预测类别和置信度

    Args:
        args: 模型配置参数
        model: 已加载的模型
        ART_feature: ART特征数据 (numpy array)
        DT_feature: DT特征数据 (numpy array)
        ERT_feature: ERT特征数据 (numpy array)
        RT_feature: RT特征数据 (numpy array)
        RDT_feature: RDT特征数据 (numpy array)
        gesture_dict: 手势类别字典，用于将预测索引转换为手势名称

    Returns:
        predicted_class: 预测的类别索引
        predicted_gesture: 预测的手势名称 (如果提供了gesture_dict)
        confidence_score: 预测置信度分数
        raw_outputs: 原始模型输出
    """
    # 调用基础推理函数
    outputs, confidence = realtime_inference(args, model, ART_feature, DT_feature, ERT_feature, RT_feature, RDT_feature)

    # 获取预测类别
    if args.method == 'DScombine':
        # 对于DScombine方法，outputs是alpha_combined - 1
        predicted_class = torch.argmax(outputs, dim=-1).cpu().numpy()
        confidence_score = torch.max(torch.softmax(outputs, dim=-1), dim=-1)[0].cpu().numpy()
    else:
        # 对于其他方法，outputs是logits
        predicted_class = torch.argmax(outputs, dim=-1).cpu().numpy()
        confidence_score = torch.max(torch.softmax(outputs, dim=-1), dim=-1)[0].cpu().numpy()

    # 由于我们添加了batch维度(batch_size=1)，需要取第一个元素
    if isinstance(predicted_class, np.ndarray) and predicted_class.shape == (1,):
        predicted_class = predicted_class[0]
    if isinstance(confidence_score, np.ndarray) and confidence_score.shape == (1,):
        confidence_score = confidence_score[0]

    # 转换为手势名称
    predicted_gesture = None
    if gesture_dict is not None:
        predicted_gesture = gesture_dict.get(str(predicted_class), f"Unknown_{predicted_class}")

    return predicted_class, predicted_gesture, confidence_score, outputs


def simple_realtime_inference(model, args, RT_feature, DT_feature, RDT_feature, ART_feature, ERT_feature, gesture_dict=None, confidence_threshold=0.5):
    """
    简化的实时推理函数，可以直接替换Judge_gesture函数

    Args:
        model: 已加载的模型
        args: 模型配置参数
        RT_feature: RT特征数据
        DT_feature: DT特征数据
        RDT_feature: RDT特征数据
        ART_feature: ART特征数据
        ERT_feature: ERT特征数据
        gesture_dict: 手势类别字典
        confidence_threshold: 置信度阈值，低于此值返回'NO'

    Returns:
        predicted_gesture: 预测的手势名称
    """
    if model is None:
        return "NO"

    try:
        # 执行推理
        predicted_class, predicted_gesture, confidence_score, _ = realtime_inference_with_prediction(
            args, model, ART_feature, DT_feature, ERT_feature, RT_feature, RDT_feature, gesture_dict
        )

        # 置信度检查
        if confidence_score < confidence_threshold:
            return "NO"

        return predicted_gesture

    except Exception as e:
        print(f"推理错误: {e}")
        return "NO"


def test_batch_dimensions():
    """
    测试函数，验证batch维度是否正确处理
    """
    print("=== 测试batch维度处理 ===")

    # 模拟特征数据
    ART_feature = np.random.randn(64, 64, 64).astype(np.float32)  # 3D特征
    DT_feature = np.random.randn(64, 64).astype(np.float32)       # 2D特征
    ERT_feature = np.random.randn(64, 64, 64).astype(np.float32)  # 3D特征
    RT_feature = np.random.randn(64, 64).astype(np.float32)       # 2D特征
    RDT_feature = np.random.randn(64, 64, 64).astype(np.float32)  # 3D特征

    print("原始特征形状:")
    print(f"ART_feature: {ART_feature.shape}")
    print(f"DT_feature: {DT_feature.shape}")
    print(f"ERT_feature: {ERT_feature.shape}")
    print(f"RT_feature: {RT_feature.shape}")
    print(f"RDT_feature: {RDT_feature.shape}")

    # 使用data_add_channel处理
    ART_processed, DT_processed, ERT_processed, RT_processed, RDT_processed = data_add_channel(
        ART_feature, DT_feature, ERT_feature, RT_feature, RDT_feature
    )

    print("\ndata_add_channel处理后的形状:")
    print(f"ART_processed: {ART_processed.shape}")
    print(f"DT_processed: {DT_processed.shape}")
    print(f"ERT_processed: {ERT_processed.shape}")
    print(f"RT_processed: {RT_processed.shape}")
    print(f"RDT_processed: {RDT_processed.shape}")

    # 转换为torch张量并添加batch维度
    device = torch.device('cpu')
    feature_tensors = {
        'ART': torch.from_numpy(ART_processed).float().unsqueeze(0).to(device),
        'DT': torch.from_numpy(DT_processed).float().unsqueeze(0).to(device),
        'ERT': torch.from_numpy(ERT_processed).float().unsqueeze(0).to(device),
        'RT': torch.from_numpy(RT_processed).float().unsqueeze(0).to(device),
        'RDT': torch.from_numpy(RDT_processed).float().unsqueeze(0).to(device)
    }

    print("\n添加batch维度后的torch张量形状:")
    for name, tensor in feature_tensors.items():
        print(f"{name}: {tensor.shape}")

    print("\n✓ batch维度处理正确！")


def test_yaml_loading():
    """
    测试YAML配置加载功能
    """
    print("\n=== 测试YAML配置加载 ===")

    yaml_path = "save_model/2025-07-11_11-49-14/output/args.yaml"

    try:
        # 加载配置
        args = load_args_from_yaml(yaml_path)

        print("\n加载的配置参数:")
        print(f"  backbone: {args.backbone}")
        print(f"  cnn_output_size: {args.cnn_output_size}")
        print(f"  hidden_size: {args.hidden_size}")
        print(f"  fusion_mode: {args.fusion_mode}")
        print(f"  method: {args.method}")
        print(f"  selected_features: {args.selected_features}")
        print(f"  input_feature_shapes:")
        for feature, shape in args.input_feature_shapes.items():
            print(f"    {feature}: {shape}")

        print(f"\n类别映射:")
        for class_id, class_name in args.class_mapping.items():
            print(f"    {class_id}: {class_name}")

        print("\n✅ YAML配置加载测试通过！")
        return args

    except Exception as e:
        print(f"❌ YAML配置加载失败: {e}")
        return None


if __name__ == "__main__":
    # 测试batch维度处理
    test_batch_dimensions()

    # 测试YAML配置加载
    test_yaml_loading()

