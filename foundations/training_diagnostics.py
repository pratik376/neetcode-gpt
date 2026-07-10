import torch
import torch.nn as nn
from typing import List, Dict



class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.

        answer=[]
        
        with torch.no_grad():
            current_input= x

            for layer in model:
                model_dict= {}


                if isinstance(layer,nn.Linear):

                    output= layer(current_input)
                    
                    model_dict['mean']= round(output.mean().item(),4)
                    model_dict['std']= round(output.std().item(),4)
                    dead_neurons = (output <= 0).all(dim=0)
                    dead_fraction = dead_neurons.float().mean()
                    model_dict['dead_fraction']= round(dead_fraction.item(),4)

                    answer.append(model_dict)
                else:
                    output= layer(current_input)
                
                current_input= output       

        return answer
  


    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:

        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        
        model.zero_grad()
        output= model(x)
        criterion = nn.MSELoss()
        loss= criterion(output, y)
        loss.backward()
        answer=[]

        for layer in model:

            grad_dict={}

            if isinstance(layer, nn.Linear):

                grad= layer.weight.grad

                grad_dict['mean']= round(grad.mean().item(),4)
                grad_dict['norm']=round(torch.norm(grad).item(),4)
                grad_dict['std']=round(grad.std().item(),4)
                answer.append(grad_dict)
        return answer

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
    # 1. Dead neurons
        for layer in activation_stats:
            if layer["dead_fraction"] > 0.5:
                return "dead_neurons"

        # 2. Exploding gradients
        for layer in gradient_stats:
            if layer["norm"] > 1000:
                return "exploding_gradients"

        # 3. Vanishing gradients (last layer)
        if gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        # 4. Activation standard deviation
        for layer in activation_stats:
            if layer["std"] < 0.1:
                return "vanishing_gradients"
            if layer["std"] > 10.0:
                return "exploding_gradients"

        # 5. Everything looks good
        return "healthy"






































        
