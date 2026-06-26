import torch

def evaluate(model, test_loader, loss_fn):
    """Evaluate on test set"""
    print(f"\n{'='*60}")
    print(f"EVALUATION")
    print(f"{'='*60}\n")
    
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_test, Y_test in test_loader:
            y_pred = model(X_test)
            loss, _ = loss_fn(y_pred, Y_test, 0, is_stage1=True)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"Evaluation complete!")