"""
Parse TensorBoard logs and print results
"""

from tensorboard.backend.event_processing import event_accumulator
import os

def parse_logs(log_dir):
    """Parse TensorBoard logs"""
    
    # Find events file
    event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out')]
    if not event_files:
        print("No log files found!")
        return
    
    event_file = os.path.join(log_dir, event_files[0])
    
    # Load events
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    print("="*60)
    print("TRAINING LOG SUMMARY")
    print("="*60)
    
    # Get all scalar tags
    tags = ea.Tags()['scalars']
    
    for tag in sorted(tags):
        events = ea.Scalars(tag)
        
        if len(events) == 0:
            continue
        
        # Get last value
        last_value = events[-1].value
        last_step = events[-1].step
        
        # Get max value
        max_value = max([e.value for e in events])
        max_step = [e.step for e in events if e.value == max_value][0]
        
        print(f"\n{tag}:")
        print(f"  Last (epoch {last_step}): {last_value:.4f}")
        print(f"  Best (epoch {max_step}): {max_value:.4f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    parse_logs('checkpoints/stage1/logs')