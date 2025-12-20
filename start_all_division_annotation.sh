#!/bin/bash
# Start all division annotations in order: 4 -> 3 -> 2 -> 1

echo "Starting division annotation pipeline..."
echo "Order: Type 4 -> 3 -> 2 -> 1"
echo ""

# Create monitoring script
cat > monitor_division_annotation.sh << 'MONITOR'
#!/bin/bash
echo "=== Division Annotation Progress ==="
echo ""
for type in type4_word_reasoning type3_word type2_text_reasoning type1_text; do
    if [ -f "data/division_added/$type/train.jsonl" ]; then
        count=$(wc -l < "data/division_added/$type/train.jsonl")
        echo "$type: $count samples"
    else
        echo "$type: Not started"
    fi
done
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader
echo ""
echo "Running processes:"
ps aux | grep fast_division | grep -v grep || echo "None"
MONITOR
chmod +x monitor_division_annotation.sh

# Type 4 - WORD_REASONING (smallest)
echo "=== Starting Type 4: WORD_REASONING (7,957 samples) ==="
nohup python3 scripts/fast_division_annotation.py \
    --type type4_word_reasoning \
    --model deepseek-ai/deepseek-llm-7b-chat \
    --device cuda:1 \
    --batch-size 4 \
    > logs/fast_division_type4.log 2>&1 &
PID4=$!
echo "Type 4 started. PID: $PID4"
echo ""

# Wait for Type 4 to complete
echo "Waiting for Type 4 to complete (~2.2 hours)..."
while kill -0 $PID4 2>/dev/null; do
    sleep 300  # Check every 5 minutes
done
echo "✓ Type 4 complete"
echo ""

# Type 3 - WORD (MCQ)
echo "=== Starting Type 3: WORD (16,701 samples) ==="
nohup python3 scripts/fast_division_annotation.py \
    --type type3_word \
    --model deepseek-ai/deepseek-llm-7b-chat \
    --device cuda:1 \
    --batch-size 8 \
    > logs/fast_division_type3.log 2>&1 &
PID3=$!
echo "Type 3 started. PID: $PID3"
echo ""

# Wait for Type 3 to complete
echo "Waiting for Type 3 to complete (~1.8 hours)..."
while kill -0 $PID3 2>/dev/null; do
    sleep 300
done
echo "✓ Type 3 complete"
echo ""

# Type 2 - TEXT_REASONING
echo "=== Starting Type 2: TEXT_REASONING (23,018 samples) ==="
nohup python3 scripts/fast_division_annotation.py \
    --type type2_text_reasoning \
    --model deepseek-ai/deepseek-llm-7b-chat \
    --device cuda:1 \
    --batch-size 4 \
    > logs/fast_division_type2.log 2>&1 &
PID2=$!
echo "Type 2 started. PID: $PID2"
echo ""

# Wait for Type 2 to complete
echo "Waiting for Type 2 to complete (~6.4 hours)..."
while kill -0 $PID2 2>/dev/null; do
    sleep 300
done
echo "✓ Type 2 complete"
echo ""

# Type 1 - TEXT (largest)
echo "=== Starting Type 1: TEXT (118,431 samples) ==="
# Clear old Type 1 data
rm -f data/division_added/type1_text/train.jsonl
nohup python3 scripts/fast_division_annotation.py \
    --type type1_text \
    --model deepseek-ai/deepseek-llm-7b-chat \
    --device cuda:1 \
    --batch-size 8 \
    > logs/fast_division_type1.log 2>&1 &
PID1=$!
echo "Type 1 started. PID: $PID1"
echo ""

echo "All types started sequentially."
echo "Monitor with: ./monitor_division_annotation.sh"
echo "Total ETA: ~38 hours"
echo ""
echo "Timeline:"
echo "  Now + 2.2h  - Type 4 complete"
echo "  Now + 4.0h  - Type 3 complete"
echo "  Now + 10.4h - Type 2 complete"
echo "  Now + 48.4h - Type 1 complete (ALL DONE)"
