#!/bin/bash   

# ==========================================================   
# 天线基站三维建模 (V4.8 - 源码级适配版)   
# 适配 convert.py 硬编码的 "/input" 逻辑
# ==========================================================   

echo "----------------------------------------------------------"   
read -p "请输入图片目录的绝对路径: " IMAGE_SOURCE  
read -p "请输入项目输出根目录: " PROJECT_ROOT   

# 强制规范化路径
PROJECT_ROOT=$(realpath "${PROJECT_ROOT%/}")
IMAGE_SOURCE=$(realpath "${IMAGE_SOURCE%/}")

# 1. 严格目录构造 (由于 convert.py 的硬编码，必须有 input 文件夹)
INPUT_DIR="$PROJECT_ROOT/input"
GS_MODEL_OUT="$PROJECT_ROOT/gs_training"
SUGAR_COARSE_OUT="$PROJECT_ROOT/sugar_coarse"

mkdir -p "$INPUT_DIR"
mkdir -p "$GS_MODEL_OUT"
mkdir -p "$SUGAR_COARSE_OUT"

# --- [STEP 1] 同步图片到 input (这是 convert.py 唯一认识的地方) ---
echo "[STEP 1] 正在同步图片到 $INPUT_DIR ..."
cp "$IMAGE_SOURCE"/*.{jpg,JPG,png,PNG} "$INPUT_DIR/" 2>/dev/null

# 检查同步结果
IMG_COUNT=$(ls -1 "$INPUT_DIR" | wc -l)
if [ "$IMG_COUNT" -lt 5 ]; then
    echo "❌ 严重错误: 在 $INPUT_DIR 中没找到图片！"
    exit 1
fi
echo "✅ 图片同步成功，共 $IMG_COUNT 张。"

# --- [STEP 2] COLMAP 位姿计算 ---
echo "[STEP 2] 正在运行 COLMAP..."
# 根据 convert.py 源码，-s 传根目录后，它会自动拼 /input
python gaussian_splatting/convert.py -s "$PROJECT_ROOT"

# 【安全熔断 1】检查位姿是否生成
# convert.py 会在根目录下生成 /sparse 文件夹
if [ ! -d "$PROJECT_ROOT/sparse" ]; then
    echo "❌ 严重错误: COLMAP 重建失败，未发现 $PROJECT_ROOT/sparse 文件夹！"
    exit 1
fi
echo "✅ COLMAP 重建圆满完成。"

# --- [PHASE 1] 基础 3DGS 训练 ---
echo "[PHASE 1] 正在训练基础 3DGS..."
python gaussian_splatting/train.py -s "$PROJECT_ROOT" -m "$GS_MODEL_OUT" --iterations 7000 --eval  

# 【安全熔断 2】
if [ ! -f "$GS_MODEL_OUT/cameras.json" ]; then
    echo "❌ 严重错误: 基础训练未产生 cameras.json！"
    exit 1
fi

# --- [PHASE 2] SuGaR 正则化 (7000+8000=15000) ---
echo "[PHASE 2] 正在运行 SuGaR 正则化..."
python train_coarse_density.py -s "$PROJECT_ROOT" -c "$GS_MODEL_OUT" -i 8000 -e 0.1 -n 0.1 -o "$SUGAR_COARSE_OUT"

# 【安全熔断 3】
COARSE_MODEL_PATH="$SUGAR_COARSE_OUT/15000.pt"
if [ ! -f "$COARSE_MODEL_PATH" ]; then
    echo "❌ 严重错误: 未发现 $COARSE_MODEL_PATH ！"
    exit 1
fi

# --- [PRE-EXTRACT] 临门一脚数据清洗 ---
echo "--- [PRE-EXTRACT] 正在清洗图片以备 Mesh 提取 ---"
# 注意：convert.py 运行后，模型训练读取的是根目录下的 images/ (去畸变后的图)
IMAGES_WORKING="$PROJECT_ROOT/images"
CLEAN_OUTPUT="$PROJECT_ROOT/images_clean"

python perfect_clean.py --input_dir "$IMAGES_WORKING" --output_dir "$CLEAN_OUTPUT"

if [ -d "$CLEAN_OUTPUT" ] && [ "$(ls -A "$CLEAN_OUTPUT")" ]; then
    echo "✅ 清洗完成，替换 $IMAGES_WORKING 文件夹..."
    mv "$IMAGES_WORKING" "$PROJECT_ROOT/images_original_backup"
    mv "$CLEAN_OUTPUT" "$IMAGES_WORKING"
fi

# --- [PHASE 3] 最终提取 ---
echo "[PHASE 3] 正在启动 Mesh 提取..."
python extract_mesh.py -s "$PROJECT_ROOT" -c "$GS_MODEL_OUT" -m "$COARSE_MODEL_PATH" -i 7000 -l 0.1 -d 300000

echo "🎉 全部流程已完成！"