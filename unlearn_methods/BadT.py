
import torch
import torch.nn.functional as F

def UnlearnerLoss(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
    labels = torch.unsqueeze(labels, dim = 1)
    
    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

    # label 1 means forget sample
    # label 0 means retain sample
    overall_teacher_out = labels * u_teacher_out + (1-labels)*f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out, reduction='batchmean')

def label_reconstruct(labels, unlearn_classes):
    if isinstance(unlearn_classes, (list, tuple)):
        target_reconstruct = torch.zeros_like(labels)
        for c in unlearn_classes:
            target_reconstruct[labels == c] = 1
    else:
        target_reconstruct = torch.zeros_like(labels)
        target_reconstruct[labels == unlearn_classes] = 1
    return target_reconstruct

def BadT_unlearning(full_component_teacher, unlearning_teacher, student_model,
                    training_loader, unlearn_classes, device, 
                    unlearn_lr=0.001, unlearn_epochs=10, 
                    KL_temperature=1):

    for param in full_component_teacher.parameters():
        param.requires_grad = False
    for param in unlearning_teacher.parameters():
        param.requires_grad = False
    full_component_teacher.eval()
    unlearning_teacher.eval()

    optimizer = torch.optim.SGD(student_model.parameters(), lr=unlearn_lr)
    student_model.train()
    for epoch in range(unlearn_epochs):
        for batch, (x, y) in enumerate(training_loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                unlearning_teacher_out = unlearning_teacher(x)
                full_component_teacher_out = full_component_teacher(x)
            student_out = student_model(x)
            optimizer.zero_grad()
            target_reconstruct = label_reconstruct(y, unlearn_classes).to(device)
            loss = UnlearnerLoss(output=student_out, 
                                 labels=target_reconstruct, 
                                 full_teacher_logits=full_component_teacher_out, 
                                 unlearn_teacher_logits=unlearning_teacher_out, 
                                 KL_temperature=KL_temperature)
            loss.backward()
            optimizer.step()
    return student_model
