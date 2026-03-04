import cv2
import numpy as np
import svgwrite
import base64
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import math

class InteractiveEdgeEditor:
    def __init__(self, image_path='exp.png'):
        self.root = tk.Tk()
        self.root.title("🎛️ DUAL PREVIEW 50% - NODE EDITOR + WELD PREVIEW")
        self.root.geometry("1800x1000")
        
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        if self.img is None:
            print(f"❌ Không tìm thấy {image_path}")
            return
        self.height, self.width = self.img.shape[:2]
        
        self.params = {
            'canny_low': 50, 'canny_high': 100, 'contour_area': 250,
            'epsilon_factor': 0.01, 'kernel_size': 5, 'sobel_ksize': 1
        }
        
        self.node_positions = {}
        self.line_nodes = {}
        self.canvas_scale_x = 600 / self.width
        self.canvas_scale_y = 350 / self.height
        self.weld_threshold = 25
        
        self.selected_node = None
        self.dragging = False
        self.value_labels = {}
        
        self.setup_gui()
        self.update_previews()
    
    def setup_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_frame = ttk.LabelFrame(main_frame, text="🎛️ CONTROLS + WELD/DELETE", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        
        slider_configs = [
            ("Canny Low", 25, 100, self.params['canny_low']),
            ("Canny High", 80, 250, self.params['canny_high']),
            ("Contour Area", 50, 500, self.params['contour_area']),
            ("Epsilon", 0.005, 0.03, self.params['epsilon_factor']),
            ("Kernel Size", 1, 7, self.params['kernel_size']),
            ("Sobel Ksize", 1, 7, self.params['sobel_ksize'])
        ]
        
        for label_text, minv, maxv, default in slider_configs:
            row_frame = ttk.Frame(control_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(row_frame, text=label_text, width=12).pack(side=tk.LEFT)
            scale_var = tk.DoubleVar(value=default)
            scale = ttk.Scale(row_frame, from_=minv, to=maxv, variable=scale_var,
                            command=self.on_slider_change)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5,5))
            
            value_label = ttk.Label(row_frame, text=f"{default:.3f}", width=8)
            value_label.pack(side=tk.RIGHT)
            self.value_labels[label_text] = (scale_var, value_label)
        
        ttk.Separator(control_frame).pack(fill=tk.X, pady=10)
        ttk.Label(control_frame, text="🔗 WELD NODES", font=('bold', 10)).pack(anchor=tk.W)
        
        weld_row = ttk.Frame(control_frame)
        weld_row.pack(fill=tk.X, pady=2)
        ttk.Label(weld_row, text="Weld Dist", width=12).pack(side=tk.LEFT)
        
        self.weld_var = tk.DoubleVar(value=self.weld_threshold)
        weld_scale = ttk.Scale(weld_row, from_=5, to=60, variable=self.weld_var,
                             command=self.on_weld_change)
        weld_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5,5))
        
        self.weld_label = ttk.Label(weld_row, text=f"{self.weld_threshold}px", width=8)
        self.weld_label.pack(side=tk.RIGHT)
        
        ttk.Separator(control_frame).pack(fill=tk.X, pady=10)
        ttk.Button(control_frame, text="🔄 REFRESH", command=self.update_previews).pack(fill=tk.X)
        ttk.Button(control_frame, text="🔗 WELD ALL", command=self.weld_all_nodes).pack(fill=tk.X)
        ttk.Button(control_frame, text="🗑️ DELETE SELECTED", command=self.delete_selected).pack(fill=tk.X)
        ttk.Button(control_frame, text="💾 EXPORT SVG", command=self.export_svg).pack(fill=tk.X)
        
        self.status_label = ttk.Label(control_frame, text="📐 Ready")
        self.status_label.pack(pady=10)
        
        preview_frame = ttk.LabelFrame(main_frame, text="👁️ DUAL PREVIEW (50%)", padding=5)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        left_frame = ttk.Frame(preview_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5))
        ttk.Label(left_frame, text="✏️ NODE EDITOR (Drag=🖱️ Del=DEL)", font=('bold', 10)).pack()
        self.canvas_edit = tk.Canvas(left_frame, bg='black', width=600, height=350, relief='solid', bd=1)
        self.canvas_edit.pack(fill=tk.BOTH, expand=True)
        
        right_frame = ttk.Frame(preview_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5,0))
        ttk.Label(right_frame, text="🔗 WELD PREVIEW (Realtime)", font=('bold', 10)).pack()
        self.canvas_weld = tk.Canvas(right_frame, bg='black', width=600, height=350, relief='solid', bd=1)
        self.canvas_weld.pack(fill=tk.BOTH, expand=True)
        
        self.canvas_edit.bind("<Button-1>", self.on_click)
        self.canvas_edit.bind("<B1-Motion>", self.on_drag)
        self.canvas_edit.bind("<ButtonRelease-1>", self.on_release)
        self.canvas_edit.bind("<Delete>", self.on_delete_key)
        self.canvas_edit.focus_set()
    
    def on_slider_change(self, event):
        for label_text, (scale_var, value_label) in self.value_labels.items():
            value_label.config(text=f"{scale_var.get():.3f}")
        self.root.after(300, self.update_previews)
    
    def on_weld_change(self, value):
        self.weld_threshold = self.weld_var.get()
        self.weld_label.config(text=f"{self.weld_threshold:.0f}px")
        self.draw_weld_preview()
    
    def get_params(self):
        params = {k.lower().replace(" ", "_"): v[0].get() for k, v in self.value_labels.items()}
        return params
    
    def process_image(self):
        params = self.get_params()
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        sobel_ksize = int(params.get('sobel_ksize', 3))
        if sobel_ksize % 2 == 0: sobel_ksize += 1
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel_norm = np.uint8(255 * sobel / np.max(sobel))
        
        edges = cv2.Canny(sobel_norm, int(params['canny_low']), int(params['canny_high']))
        kernel_size = int(params.get('kernel_size', 2))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_lines = []
        
        for contour in contours:
            if cv2.contourArea(contour) > params['contour_area']:
                epsilon = params['epsilon'] * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = approx.reshape(-1, 2).tolist()
                if len(points) >= 3:
                    raw_lines.append(points)
        
        return raw_lines
    
    def update_previews(self):
        if not self.line_nodes:
            self.init_nodes_from_detection()
        
        self.status_label.config(text=f"📐 Nodes: {len(self.node_positions)} | Welded: {self.get_welded_count()}")
        self.draw_node_editor()
        self.draw_weld_preview()
    
    def init_nodes_from_detection(self):
        raw_lines = self.process_image()
        self.line_nodes = {}
        node_counter = 0
        
        for line_idx, points in enumerate(raw_lines[:8]):
            line_id = f"line_{line_idx}"
            node_ids = []
            
            for point in points:
                node_id = f"node_{node_counter}"
                self.node_positions[node_id] = point
                node_ids.append(node_id)
                node_counter += 1
            
            self.line_nodes[line_id] = node_ids
    
    def draw_background(self, canvas):
        bg = cv2.resize(self.img, (600, 350))
        bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(bg_rgb))
        canvas.create_image(0, 0, anchor=tk.NW, image=photo, tags="bg")
        canvas.image = photo
    
    def draw_node_editor(self):
        self.canvas_edit.delete("all")
        self.draw_background(self.canvas_edit)
        
        scale_x, scale_y = self.canvas_scale_x, self.canvas_scale_y
        
        for line_id, node_ids in self.line_nodes.items():
            canvas_points = []
            for node_id in node_ids:
                if node_id in self.node_positions:
                    x, y = self.node_positions[node_id]
                    canvas_points.extend([x*scale_x, y*scale_y])
            
            if len(canvas_points) >= 6:
                self.canvas_edit.create_polygon(canvas_points, outline='#FFAA00', 
                                              width=2, fill='', tags=(line_id, "line"))
            
            for node_id in node_ids:
                if node_id in self.node_positions:
                    x, y = self.node_positions[node_id]
                    canvas_x, canvas_y = x*scale_x, y*scale_y
                    
                    fill_color = '#FFAA00' if node_id == self.selected_node else '#00FF00'
                    self.canvas_edit.create_oval(canvas_x-10, canvas_y-10, canvas_x+10, canvas_y+10,
                                               fill=fill_color, outline='#00AA00', width=3,
                                               tags=(node_id, "node"))
    
    def get_welded_positions(self):
        welded_nodes = {}
        node_groups = {}
        
        for node_id, pos in self.node_positions.items():
            if node_id in welded_nodes:
                continue
                
            px, py = pos
            group = [node_id]
            
            for other_id, other_pos in self.node_positions.items():
                if other_id == node_id or other_id in group:
                    continue
                    
                ox, oy = other_pos
                dist = math.sqrt((px-ox)**2 + (py-oy)**2)
                
                if dist < self.weld_threshold:
                    group.append(other_id)
            
            avg_x = sum(self.node_positions[nid][0] for nid in group) / len(group)
            avg_y = sum(self.node_positions[nid][1] for nid in group) / len(group)
            
            rep_id = node_id
            node_groups[rep_id] = group
            welded_nodes[rep_id] = [avg_x, avg_y]
        
        return welded_nodes, node_groups
    
    def merge_lines_by_welded_nodes(self, node_groups, welded_positions):
        all_welded_points = []
        
        for line_id, node_ids in self.line_nodes.items():
            line_points = []
            for node_id in node_ids:
                if node_id in self.node_positions:
                    for rep_id, group in node_groups.items():
                        if node_id in group:
                            line_points.append(welded_positions[rep_id])
                            break
                    else:
                        line_points.append(self.node_positions[node_id])
            
            if len(line_points) >= 3:
                all_welded_points.extend(line_points)
        
        unique_points = self.remove_duplicate_points(all_welded_points, self.weld_threshold * 0.5)
        return [unique_points]
    
    def remove_duplicate_points(self, points, threshold):
        kept_points = []
        for point in points:
            is_duplicate = False
            for kept in kept_points:
                dist = math.sqrt((point[0]-kept[0])**2 + (point[1]-kept[1])**2)
                if dist < threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept_points.append(point)
        return kept_points
    
    def draw_weld_preview(self):
        self.canvas_weld.delete("all")
        self.draw_background(self.canvas_weld)
        
        welded_positions, node_groups = self.get_welded_positions()
        merged_lines = self.merge_lines_by_welded_nodes(node_groups, welded_positions)
        
        scale_x, scale_y = self.canvas_scale_x, self.canvas_scale_y
        for line_points in merged_lines:
            # 🔥 FIX SYNTAX ERROR
            canvas_points = []
            for p in line_points:
                canvas_points.extend([p[0]*scale_x, p[1]*scale_y])
            
            if len(canvas_points) >= 6:
                self.canvas_weld.create_polygon(canvas_points, outline='#FF0000', 
                                              width=5, fill='', tags="welded_line")
    
    def get_welded_count(self):
        welded, _ = self.get_welded_positions()
        return len(welded)
    
    def on_drag(self, event):
        if self.dragging and self.selected_node:
            real_x = event.x / self.canvas_scale_x
            real_y = event.y / self.canvas_scale_y
            self.node_positions[self.selected_node] = [real_x, real_y]
            
            self.draw_node_editor()
            self.draw_weld_preview()
    
    def weld_all_nodes(self):
        welded_positions, node_groups = self.get_welded_positions()
        
        new_line_nodes = {}
        merged_points = []
        
        for rep_id, group in node_groups.items():
            merged_points.append(welded_positions[rep_id])
        
        if len(merged_points) >= 3:
            new_line_id = "merged_line"
            new_node_ids = []
            for i, point in enumerate(merged_points):
                new_node_id = f"merged_node_{i}"
                self.node_positions[new_node_id] = point
                new_node_ids.append(new_node_id)
            new_line_nodes[new_line_id] = new_node_ids
        
        self.line_nodes = new_line_nodes
        self.update_previews()
        print(f"🔗 ✅ WELD APPLIED: {len(merged_points)} merged nodes → 1 line!")
    
    def delete_selected(self):
        if self.selected_node and self.selected_node in self.node_positions:
            del self.node_positions[self.selected_node]
            
            for line_id, node_ids in list(self.line_nodes.items()):
                if self.selected_node in node_ids:
                    node_ids.remove(self.selected_node)
                    if len(node_ids) < 3:
                        del self.line_nodes[line_id]
            
            self.selected_node = None
            self.update_previews()
    
    def on_delete_key(self, event):
        self.delete_selected()
    
    def on_click(self, event):
        items = self.canvas_edit.find_overlapping(event.x-10, event.y-10, event.x+10, event.y+10)
        for item in items:
            tags = self.canvas_edit.gettags(item)
            for tag in tags:
                if tag.startswith("node_") or tag.startswith("merged_node_"):
                    self.selected_node = tag
                    self.dragging = True
                    return
    
    def on_release(self, event):
        self.dragging = False
    
    def export_svg(self):
        dwg = svgwrite.Drawing('bien_gioi_dual.svg', size=(f'{self.width}px', f'{self.height}px'))
        with open(self.image_path, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode()
        dwg.add(dwg.image(href=f'data:image/jpeg;base64,{img_b64}', insert=(0,0),
                         size=(self.width, self.height), opacity=0.85))
        
        welded_positions, _ = self.get_welded_positions()
        for line_id, node_ids in self.line_nodes.items():
            points = [welded_positions.get(nid, self.node_positions.get(nid, [0,0])) for nid in node_ids if nid in self.node_positions]
            if len(points) >= 3:
                path_data = f"M{points[0][0]:.1f},{points[0][1]:.1f} "
                for p in points[1:]:
                    path_data += f"L{p[0]:.1f},{p[1]:.1f} "
                path_data += "Z"
                dwg.add(dwg.path(d=path_data, stroke='#FF0000', stroke_width=8,
                               stroke_linecap='round', fill='none'))
        
        dwg.save()
        print("✅ EXPORT: bien_gioi_dual.svg")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = InteractiveEdgeEditor('exp.png')
    app.run()
