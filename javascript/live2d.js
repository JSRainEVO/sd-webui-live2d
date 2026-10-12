function switch_to_live2d() {
    const root = gradioApp();
    const target = root.querySelector('#tab_live2d');
    if (!target) return [];
    const panels = Array.from(root.querySelectorAll('#tabs > .tabitem'));
    const index = panels.indexOf(target);
    const buttons = root.querySelectorAll('#tabs > .tab-nav > button');
    if (index >= 0 && buttons[index]) buttons[index].click();
    return [];
}

function live2d_selected_result(gallery, info, placeholder) {
    const chosen = extract_image_from_gallery(gallery)[0];
    const index = Math.max(0, selected_gallery_index());
    return [chosen, info, index];
}
