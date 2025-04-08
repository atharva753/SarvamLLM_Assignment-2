import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any


def parse_pattern(pattern: str) -> Tuple[str, str]:
    """
    Parse the pattern string and split it into input and output parts.
    
    Args:
        pattern: The einops pattern string (e.g., 'b c h w -> b h w c')
        
    Returns:
        A tuple of (input_pattern, output_pattern)
    """
    if '->' not in pattern:
        raise ValueError("Pattern must contain '->' to separate input and output dimensions")
    
    parts = pattern.split('->')
    if len(parts) != 2:
        raise ValueError("Pattern must contain exactly one '->'")
    
    input_pattern = parts[0].strip()
    output_pattern = parts[1].strip()
    
    return input_pattern, output_pattern


def parse_axis(axis: str) -> Tuple[str, bool, List[str]]:
    """
    Parse a single axis specification, handling parentheses for merging/splitting.
    
    Args:
        axis: The axis specification (e.g., '(h w)', 'b', etc.)
        
    Returns:
        A tuple of (axis_name, is_composite, components)
        where is_composite is True if the axis is a composition (has parentheses),
        and components is a list of sub-axes names.
    """
    is_composite = False
    components = []
    
    # Check if this is a composite axis (has parentheses)
    if axis.startswith('(') and axis.endswith(')'):
        is_composite = True
        # Extract the components within parentheses
        inner_content = axis[1:-1].strip()
        components = [comp.strip() for comp in inner_content.split()]
        return axis, is_composite, components
    
    return axis, is_composite, [axis]


def parse_axes(pattern: str) -> List[Tuple[str, bool, List[str]]]:
    """
    Parse all axes in a pattern.
    
    Args:
        pattern: A pattern string (either input or output part)
        
    Returns:
        A list of parsed axes (each a tuple as returned by parse_axis)
    """
    # Handle special case of ellipsis
    if '...' in pattern:
        parts = pattern.split('...')
        if len(parts) != 2:
            raise ValueError("Pattern can contain at most one ellipsis (...)")
        
        before_ellipsis = parts[0].strip()
        after_ellipsis = parts[1].strip()
        
        before_axes = [] if not before_ellipsis else before_ellipsis.split()
        after_axes = [] if not after_ellipsis else after_ellipsis.split()
        
        parsed_before = [parse_axis(ax) for ax in before_axes]
        ellipsis_marker = [('...', False, ['...'])]
        parsed_after = [parse_axis(ax) for ax in after_axes]
        
        return parsed_before + ellipsis_marker + parsed_after
    
    # Regular case without ellipsis
    return [parse_axis(ax) for ax in pattern.split()]


def get_shape_from_axes(
    parsed_axes: List[Tuple[str, bool, List[str]]],
    tensor_shape: Tuple[int, ...],
    axes_lengths: Dict[str, int],
    is_input: bool
) -> List[int]:
    """
    Calculate the shape represented by the parsed axes.
    
    Args:
        parsed_axes: The list of parsed axes
        tensor_shape: The shape of the input tensor
        axes_lengths: Dictionary mapping axis names to their lengths
        is_input: Whether this is for input axes (True) or output axes (False)
        
    Returns:
        A list of dimension lengths
    """
    result_shape = []
    tensor_dim_idx = 0
    ellipsis_dims = []
    
    for i, (axis, is_composite, components) in enumerate(parsed_axes):
        # Handle ellipsis
        if axis == '...':
            if is_input:
                # For input, calculate how many dimensions are covered by ellipsis
                remaining_explicit_dims = sum(1 for ax, _, _ in parsed_axes[i+1:] if ax != '...')
                ellipsis_size = len(tensor_shape) - (len(parsed_axes) - 1)  # -1 for the ellipsis itself
                
                # Add all ellipsis dimensions
                for j in range(ellipsis_size):
                    ellipsis_dims.append(tensor_shape[tensor_dim_idx])
                    tensor_dim_idx += 1
                
                result_shape.extend(ellipsis_dims)
            else:
                # For output, just use the same ellipsis dimensions as input
                result_shape.extend(ellipsis_dims)
            continue
        
        if is_input:
            if is_composite:
                # For composite input axes, calculate the product of component dimensions
                if axis in axes_lengths:
                    # If the composite dimension size is provided directly
                    dim_size = axes_lengths[axis]
                else:
                    # Otherwise use the tensor's actual dimension
                    dim_size = tensor_shape[tensor_dim_idx]
                
                # Verify the dimension size matches the product of component sizes
                component_product = 1
                for comp in components:
                    if comp in axes_lengths:
                        component_product *= axes_lengths[comp]
                    else:
                        raise ValueError(f"Size for component axis '{comp}' not provided")
                
                if dim_size % component_product != 0:
                    raise ValueError(
                        f"Dimension size {dim_size} is not divisible by product of component sizes {component_product}"
                    )
                
                result_shape.append(dim_size)
                tensor_dim_idx += 1
            else:
                # For simple input axes, use the tensor's dimension
                if tensor_dim_idx < len(tensor_shape):
                    result_shape.append(tensor_shape[tensor_dim_idx])
                    tensor_dim_idx += 1
                else:
                    raise ValueError(f"Input pattern has more dimensions than tensor shape {tensor_shape}")
        else:  # Output shape calculation
            if is_composite:
                # For composite output axes, calculate the product of component dimensions
                component_product = 1
                for comp in components:
                    if comp in axes_lengths:
                        component_product *= axes_lengths[comp]
                    else:
                        # Look for the component in the input dimensions
                        # (This is simplified; a more robust implementation would track all dimensions)
                        raise ValueError(f"Size for output component axis '{comp}' not provided")
                
                result_shape.append(component_product)
            else:
                # For simple output axes
                if axis in axes_lengths:
                    result_shape.append(axes_lengths[axis])
                else:
                    # For dimensions that should be preserved from input, look it up
                    # (This is simplified; a more robust implementation would track dimensions better)
                    raise ValueError(f"Size for output axis '{axis}' not provided and not found in input")
    
    return result_shape


def compute_output_shape(
    input_axes: List[Tuple[str, bool, List[str]]],
    output_axes: List[Tuple[str, bool, List[str]]],
    tensor_shape: Tuple[int, ...],
    axes_lengths: Dict[str, int]
) -> List[int]:
    """
    Compute the shape of the output tensor.
    
    Args:
        input_axes: Parsed input axes
        output_axes: Parsed output axes
        tensor_shape: Shape of the input tensor
        axes_lengths: Dictionary mapping axis names to their lengths
        
    Returns:
        The shape of the output tensor
    """
    # Build mapping from axis name to its dimension size
    dims_dict = {}
    tensor_dim_idx = 0
    ellipsis_dims = []
    
    # First pass: directly extract dimensions from tensor shape
    for axis, is_composite, components in input_axes:
        if axis == '...':
            # Calculate how many dimensions are covered by ellipsis
            remaining_explicit_dims = sum(1 for ax, _, _ in input_axes if ax != '...')
            ellipsis_size = len(tensor_shape) - remaining_explicit_dims
            
            # Store ellipsis dimensions
            for j in range(ellipsis_size):
                ellipsis_dims.append(tensor_shape[tensor_dim_idx])
                tensor_dim_idx += 1
        elif is_composite:
            # For composite axes in input, get the total size
            total_size = tensor_shape[tensor_dim_idx]
            tensor_dim_idx += 1
            
            # Calculate sizes for individual components
            remaining_product = 1
            unknown_component = None
            
            for comp in components:
                if comp in axes_lengths:
                    remaining_product *= axes_lengths[comp]
                    dims_dict[comp] = axes_lengths[comp]
                else:
                    if unknown_component is not None:
                        raise ValueError(f"Multiple unknown components in axis {axis}")
                    unknown_component = comp
            
            # If there's one unknown component, calculate its size
            if unknown_component is not None:
                if remaining_product == 0:
                    raise ValueError(f"Cannot infer size for {unknown_component}, product of other components is 0")
                if total_size % remaining_product != 0:
                    raise ValueError(f"Total size {total_size} is not divisible by product of known components {remaining_product}")
                dims_dict[unknown_component] = total_size // remaining_product
        else:
            # For simple axes, use the corresponding tensor dimension
            dims_dict[axis] = tensor_shape[tensor_dim_idx]
            tensor_dim_idx += 1
    
    # Add explicitly provided axes lengths
    for name, size in axes_lengths.items():
        dims_dict[name] = size
    
    # Second pass: compute output shape
    output_shape = []
    for axis, is_composite, components in output_axes:
        if axis == '...':
            output_shape.extend(ellipsis_dims)
        elif is_composite:
            # For composite output axes, multiply the component sizes
            product = 1
            for comp in components:
                if comp not in dims_dict:
                    raise ValueError(f"Size for component '{comp}' in output pattern not found")
                product *= dims_dict[comp]
            output_shape.append(product)
        else:
            # For simple output axes, use the corresponding dimension
            if axis not in dims_dict:
                raise ValueError(f"Size for axis '{axis}' in output pattern not found")
            output_shape.append(dims_dict[axis])
    
    return output_shape


def rearrange(tensor: np.ndarray, pattern: str, **axes_lengths) -> np.ndarray:
    """
    Rearrange a tensor according to the given pattern, similar to einops.rearrange.
    
    Args:
        tensor: Input tensor (numpy array)
        pattern: Rearrangement pattern (e.g., 'b c h w -> b h w c')
        **axes_lengths: Named sizes for axes (e.g., h=32, w=32)
        
    Returns:
        Rearranged tensor
    """
    # Parse the pattern
    input_pattern, output_pattern = parse_pattern(pattern)
    input_axes = parse_axes(input_pattern)
    output_axes = parse_axes(output_pattern)
    
    # Validate input tensor shape against the pattern
    tensor_shape = tensor.shape
    
    # Collect all named axes for validation
    all_input_components = []
    for _, _, components in input_axes:
        all_input_components.extend(components)
    
    all_output_components = []
    for _, _, components in output_axes:
        all_output_components.extend(components)
    
    # Make sure all output components are in the input or axes_lengths
    for comp in all_output_components:
        if comp != '...' and comp not in all_input_components and comp not in axes_lengths:
            raise ValueError(f"Output component '{comp}' not found in input pattern or axes_lengths")
    
    # Compute the output shape
    output_shape = compute_output_shape(input_axes, output_axes, tensor_shape, axes_lengths)
    
    # Step 1: Build mappings between input and output axes
    input_flat_axes = []
    for _, _, components in input_axes:
        if components == ['...']:
            # Calculate ellipsis size
            remaining_explicit_dims = sum(1 for comps in input_axes if comps[2] != ['...'])
            ellipsis_size = len(tensor_shape) - remaining_explicit_dims
            input_flat_axes.extend(['...'] * ellipsis_size)
        else:
            input_flat_axes.extend(components)
    
    output_flat_axes = []
    for _, _, components in output_axes:
        if components == ['...']:
            # Use same ellipsis size as input
            ellipsis_count = input_flat_axes.count('...')
            output_flat_axes.extend(['...'] * ellipsis_count)
        else:
            output_flat_axes.extend(components)
    
    # Step 2: Handle composite axes (splitting)
    # First, reshape the tensor to split composite axes
    reshape_shape = []
    reshape_axes_map = {}  # Maps each position in reshape_shape to original axis name
    
    tensor_dim_idx = 0
    ellipsis_start_idx = None
    
    # Process input axes to build reshape shape
    for axis, is_composite, components in input_axes:
        if axis == '...':
            # Mark where ellipsis starts in reshape shape
            ellipsis_start_idx = len(reshape_shape)
            
            # Calculate ellipsis size
            remaining_explicit_dims = sum(1 for ax, _, _ in input_axes if ax != '...')
            ellipsis_size = len(tensor_shape) - remaining_explicit_dims
            
            # Add ellipsis dimensions to reshape shape
            for j in range(ellipsis_size):
                dim_size = tensor_shape[tensor_dim_idx]
                reshape_shape.append(dim_size)
                reshape_axes_map[len(reshape_shape) - 1] = f"..._{j}"
                tensor_dim_idx += 1
        elif is_composite:
            # For composite axes, split into component dimensions
            total_size = tensor_shape[tensor_dim_idx]
            tensor_dim_idx += 1
            
            # Calculate sizes for each component
            component_sizes = []
            for comp in components:
                if comp in axes_lengths:
                    component_sizes.append(axes_lengths[comp])
                else:
                    raise ValueError(f"Size for component '{comp}' not provided")
            
            # Verify the product of component sizes matches the total size
            if np.prod(component_sizes) != total_size:
                raise ValueError(
                    f"Product of component sizes {np.prod(component_sizes)} does not match "
                    f"dimension size {total_size} for axis {axis}"
                )
            
            # Add all component dimensions to reshape shape
            reshape_shape.extend(component_sizes)
            for i, comp in enumerate(components):
                reshape_axes_map[len(reshape_shape) - len(components) + i] = comp
        else:
            # For simple axes, use the tensor dimension directly
            reshape_shape.append(tensor_shape[tensor_dim_idx])
            reshape_axes_map[len(reshape_shape) - 1] = axis
            tensor_dim_idx += 1
    
    # Step 3: Reshape the tensor to split axes
    reshaped_tensor = tensor.reshape(reshape_shape)
    
    # Step 4: Prepare for transposition
    # Build mapping from axis name to its position in reshaped tensor
    axis_to_position = {}
    for pos, name in reshape_axes_map.items():
        if name.startswith('...'):
            axis_to_position[name] = pos
        else:
            axis_to_position[name] = pos
    
    # Build the transposition order
    transpose_order = []
    
    # Process output axes to build transpose order
    ellipsis_positions = []
    
    # First, identify all ellipsis positions
    for pos, name in reshape_axes_map.items():
        if name.startswith('...'):
            ellipsis_positions.append(pos)
    
    for axis, is_composite, components in output_axes:
        if axis == '...':
            # Add all ellipsis positions to transpose order
            transpose_order.extend(ellipsis_positions)
        elif is_composite:
            # For composite output axes, we'll need to collect positions of all components
            component_positions = []
            for comp in components:
                if comp not in axis_to_position:
                    raise ValueError(f"Component '{comp}' not found in reshaped tensor")
                component_positions.append(axis_to_position[comp])
            transpose_order.extend(component_positions)
        else:
            # For simple output axes
            if axis not in axis_to_position:
                raise ValueError(f"Axis '{axis}' not found in reshaped tensor")
            transpose_order.append(axis_to_position[axis])
    
    # Step 5: Transpose the tensor
    transposed_tensor = np.transpose(reshaped_tensor, transpose_order)
    
    # Step 6: Reshape to final output shape
    # Group dimensions according to output pattern
    final_shape = []
    dim_idx = 0
    
    for axis, is_composite, components in output_axes:
        if axis == '...':
            # For ellipsis, preserve all its dimensions
            ellipsis_dims_count = len(ellipsis_positions)
            final_shape.extend(transposed_tensor.shape[dim_idx:dim_idx+ellipsis_dims_count])
            dim_idx += ellipsis_dims_count
        elif is_composite:
            # For composite output axes, merge the components
            component_size = np.prod([transposed_tensor.shape[dim_idx + i] for i in range(len(components))])
            final_shape.append(int(component_size))  # Convert numpy types to int
            dim_idx += len(components)
        else:
            # For simple output axes
            final_shape.append(transposed_tensor.shape[dim_idx])
            dim_idx += 1
    
    # Final reshape to get the desired output shape
    return transposed_tensor.reshape(final_shape)


# Example test functions
def test_transpose():
    x = np.random.rand(3, 4)
    result = rearrange(x, 'h w -> w h')
    assert result.shape == (4, 3)
    assert np.allclose(result, x.T)
    print("Transpose test passed")


def test_split_axis():
    x = np.random.rand(12, 10)
    result = rearrange(x, '(h w) c -> h w c', h=3, w=4)
    assert result.shape == (3, 4, 10)
    # Verify content is preserved
    for i in range(3):
        for j in range(4):
            assert np.allclose(result[i, j], x[i*4 + j])
    print("Split axis test passed")


def test_merge_axes():
    x = np.random.rand(3, 4, 5)
    result = rearrange(x, 'a b c -> (a b) c')
    assert result.shape == (12, 5)
    # Verify content is preserved
    for i in range(3):
        for j in range(4):
            assert np.allclose(result[i*4 + j], x[i, j])
    print("Merge axes test passed")


def test_ellipsis():
    x = np.random.rand(2, 3, 4, 5)
    result = rearrange(x, '... h w -> ... (h w)')
    assert result.shape == (2, 3, 20)
    print("Ellipsis test passed")


def run_all_tests():
    test_transpose()
    test_split_axis()
    test_merge_axes()
    test_ellipsis()
    print("All tests passed!")


if __name__ == "__main__":
    run_all_tests()
