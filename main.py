from manim import *
from manim import config

# Set the background color
config.background_color = "#0D1219"
#Config for white background:
#config.background_color = WHITE
#Tex.set_default(color=BLACK)
#MathTex.set_default(color=BLACK)
#Text.set_default(color=BLACK)

def Text2(t):
    return Text(t,font="Poppins")

def addWatermark():
    watermark = ImageMobject("resources/images/University_LMU_background.png")

    # Scale and position the watermark
    watermark.to_edge(DOWN).to_edge(RIGHT)
    watermark.shift(DOWN * 0.5)
    watermark.shift(RIGHT * 0.5)
    return watermark

class VQGANStructure(Scene):
    def construct(self):
        # Colors
        encoder_input_color = WHITE
        decoder_output_color = WHITE
        encoder_color = GREEN
        z_color = BLUE
        vq_color = PURPLE
        codebook_color = BLUE_E
        decoder_color = YELLOW

        num_rows = 8
        table_data = [["a"] for _ in range(num_rows)]

        # Add the watermark to the background
        #self.add(addWatermark())

        # Create the table
        encoder_table = Table(
            table_data,
            include_outer_lines=True,
            line_config={"color": encoder_input_color},
            v_buff = 1.3
        ).shift(LEFT * 4)
        encoder_table.get_columns().set_opacity(0)
        encoder_table.scale(0.25)

        # Fade in the table
        self.play(FadeIn(encoder_table))

        self.wait(2)

        # Encoder
        encoder = Polygon(
            [-3.5, -1.5, 0], [-2, -0.5, 0], [-2, 0.5, 0], [-3.5, 1.5, 0],
            color=encoder_color, fill_opacity=0
        )
        encoder_text = Text("Encoder").scale(0.5).move_to(encoder.get_center())
        self.play(FadeIn(encoder), Write(encoder_text))

        # Latent space z
        z = Rectangle(width=0.5, height=1, color=z_color, fill_opacity=0).shift(LEFT * 1.25)
        z_text = Text("z").scale(0.5).move_to(z.get_center())
        self.play(FadeIn(z), Write(z_text))

        # Vector Quantization
        vq = Rectangle(width=2, height=1, color=vq_color, fill_opacity=0).shift(RIGHT * 0.5)
        vq_text = Paragraph('Vector','Quantization', alignment='center').scale(0.45).move_to(vq.get_center())
        self.play(FadeIn(vq), Write(vq_text))

        # Quantized latent space z'
        z_quant = Rectangle(width=0.5, height=1, color=z_color, fill_opacity=0).shift(RIGHT * 2.25)
        z_quant_text = Text("z'").scale(0.5).move_to(z_quant.get_center())
        self.play(FadeIn(z_quant), Write(z_quant_text))

        # Decoder
        decoder = Polygon(
            [3, -0.5, 0], [4.5, -1.5, 0], [4.5, 1.5, 0], [3, 0.5, 0],
            color=decoder_color, fill_opacity=0
        )
        decoder_text = Text("Decoder").scale(0.5).move_to(decoder.get_center())
        self.play(FadeIn(decoder), Write(decoder_text))

        table_data = [["a"] for _ in range(num_rows)]

        # Create the table
        decoder_table = Table(
            table_data,
            include_outer_lines=True,
            line_config={"color": decoder_output_color},
            v_buff=1.3
        ).shift(RIGHT * 5)
        decoder_table.get_columns().set_opacity(0)
        decoder_table.scale(0.25)

        self.play(FadeIn(decoder_table))

        self.wait(2)

        # Codebook
        codebook = Rectangle(width=2.5, height=1, color=codebook_color, fill_opacity=0).shift(UP * 2, RIGHT * 0.5)
        codebook_text = Text("Codebook").scale(0.5).move_to(codebook.get_center())
        self.play(FadeIn(codebook), Write(codebook_text))

        self.wait(2)

        # Load the image
        image = ImageMobject("resources/images/iris.jpg")
        image.scale(0.4).shift(LEFT * 5.5)  # Adjust the scaling and position as needed

        # Display the image
        self.play(FadeIn(image))

        # Move the image to the middle of the table and shrink it
        self.play(image.animate.move_to(encoder_table.get_center()).scale(0))
        self.remove(image)

        values = ["3.7", "2.2", "1.5", "0.5", "2.8", "4.6", "1.2", "2.2"]

        # Create text objects for the values at the center of the table
        value_texts = [Text(value).scale(0.25).move_to(encoder_table.get_center()) for value in values]

        vector_values = [[3.3], [7.5], [5.3], [1.3]]

        # Create the vector using the Matrix class
        z_vector = Matrix(vector_values, v_buff=0.5, bracket_h_buff=0.2)
        z_vector.scale(0.5)
        z_vector.move_to(z.get_center())

        # Animate the values moving to each cell simultaneously
        animations = []
        for i, value_text in enumerate(value_texts):
            cell = encoder_table.get_cell((i + 1, 1))
            animations.append(value_text.animate.move_to(cell.get_center()))
        encoder_z_arrow = Arrow(encoder.get_right(), z_vector.get_left(), buff=0.1)
        self.play(*animations, GrowArrow(encoder_z_arrow))

        # Ensure the value_texts are properly added to the table cells
        for i, value_text in enumerate(value_texts):
            cell = encoder_table.get_cell((i + 1, 1))
            value_text.move_to(cell.get_center())  # Ensure the text is correctly positioned in the cell
            self.add(value_text)
            encoder_table.add(value_text)  # Add the text to the table

        self.wait(2)

        # Create copies of the values to animate them moving somewhere else
        moving_values = [value_text.copy() for value_text in value_texts]
        new_location = z.get_center()
        move_animations = [value.animate.move_to(new_location) for value in moving_values]

        self.play(*move_animations, z_text.animate.next_to(z, DOWN))
        for value in moving_values:
            self.remove(value)

        # Smoothly replace z with z_vector
        self.play(Transform(z, z_vector))

        self.wait(2)

        zoom_group = VGroup(z_vector, encoder_table, z, encoder, z_text, encoder_text, vq, vq_text, z_quant_text, z_quant, codebook_text, codebook, encoder_z_arrow)
        zoom_center = zoom_group.get_center()

        fade_out_group = VGroup(decoder,decoder_text,decoder_table)

        # Zoom into the left part (scale and move)
        self.play(
            zoom_group.animate.scale(1.5),
            FadeOut(fade_out_group)
        )

        self.wait(2)

        self.play(
            codebook_text.animate.next_to(codebook, UP)
        )

        self.wait(2)

        # Define the 4 vectors with arbitrary values
        vector1 = [[1.1], [2.2], [3.3], [4.4]]
        vector2 = [[2.2], [3.3], [4.4], [5.5]]
        vector3 = [[3.3], [4.4], [5.5], [6.6]]
        vector4 = [[4.4], [5.5], [6.6], [7.7]]

        # Create the vectors using the Matrix class
        matrix1 = Matrix(vector1, v_buff=0.5, bracket_h_buff=0.2).scale(0.5)
        matrix2 = Matrix(vector2, v_buff=0.5, bracket_h_buff=0.2).scale(0.5)
        matrix3 = Matrix(vector3, v_buff=0.5, bracket_h_buff=0.2).scale(0.5)
        matrix4 = Matrix(vector4, v_buff=0.5, bracket_h_buff=0.2).scale(0.5)

        # Arrange the vectors next to each other
        vectors = VGroup(matrix1, matrix2, matrix3, matrix4).arrange(RIGHT, buff=0.2)

        vectors.move_to(codebook.get_center())

        # Add the vectors to the scene
        self.play(FadeIn(vectors))

        self.wait(2)

        z_vq_arrow = Arrow(z.get_right(), vq.get_left(), buff=0.1)
        self.play(GrowArrow(z_vq_arrow))

        self.wait(1)

        vq_matrix1__arrow = Arrow(vq.get_top(), matrix1.get_bottom(), buff=0.1)
        self.play(GrowArrow(vq_matrix1__arrow))

        self.wait(2)

        self.play(z_quant_text.animate.next_to(z_quant, DOWN))
        self.wait(2)

        matrix1_copy = matrix1.copy()
        zoom_group.add(matrix1_copy)

        self.play(matrix1_copy.animate.move_to(z_quant.get_center()).scale(1.5), FadeOut(z_quant))
        vq_z_quant_arrow = Arrow(vq.get_right(), matrix1_copy.get_left(), buff=0.1)
        self.play(GrowArrow(vq_z_quant_arrow))
        self.wait(2)

        self.play(FadeOut(vq_matrix1__arrow))
        zoom_group.add(vectors, vq_z_quant_arrow, z_vq_arrow)

        self.remove(z_quant)
        zoom_group.remove(z_quant)
        self.wait(2)
        self.play(zoom_group.animate.scale(1/1.5).move_to(zoom_center + (codebook_text.get_top() - codebook_text.get_bottom())/2), FadeIn(fade_out_group))

        self.wait(2)
        matrix1_decoder_arrow = Arrow(matrix1_copy.get_right(), decoder.get_left(), buff=0.1)
        self.play(GrowArrow(matrix1_decoder_arrow))
        self.wait(2)

        #matrix1_copy_copy = matrix1_copy.copy()
        #self.play(matrix1_copy_copy.animate.move_to(decoder_table.get_center()).scale(0))
        #self.remove(matrix1_copy_copy)

        values = ["3.9", "2.5", "1.3", "0.4", "3.4", "3.9", "1.2", "2.0"]

        # Create text objects for the values at the center of the table
        value_texts = [Text(value).scale(0.25).move_to(matrix1_copy.get_center()) for value in values]

        # Animate the values moving to each cell simultaneously
        animations = []
        for i, value_text in enumerate(value_texts):
            cell = decoder_table.get_cell((i + 1, 1))
            animations.append(value_text.animate.move_to(cell.get_center()))
        self.play(*animations)

        # Ensure the value_texts are properly added to the table cells
        for i, value_text in enumerate(value_texts):
            cell = decoder_table.get_cell((i + 1, 1))
            value_text.move_to(cell.get_center())  # Ensure the text is correctly positioned in the cell
            self.add(value_text)
            decoder_table.add(value_text)  # Add the text to the table

        self.wait(2)


import random

from manim import *
import random

class PixelGrid(VGroup):
    def __init__(self, rows=4, cols=4, cell_size=1, colors=None, **kwargs):
        super().__init__(**kwargs)
        self.colors = colors or self.generate_random_colors(rows, cols)

        # Create a grid of squares with specified colors
        for i in range(rows):
            for j in range(cols):
                # Create a square
                square = Square(side_length=cell_size)
                # Set the position of the square
                square.move_to(np.array([j * cell_size, -i * cell_size, 0]))
                # Set the color of the square
                square.set_fill(self.colors[i][j], opacity=1.0)
                square.set_stroke(width=0)  # Remove the stroke
                # Add the square to the grid
                self.add(square)

        # Center the grid
        self.move_to(ORIGIN)

    @staticmethod
    def generate_random_colors(rows, cols):
        return [[PixelGrid.random_bright_color() for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def random_bright_color():
        return "#" + ''.join([random.choice('3456789ABCDEF') for _ in range(6)])

    @staticmethod
    def blend_with_noise(hex_color, blend_factor=0.6):
        rgb = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
        gray = int(sum(rgb) / 3)
        blended_rgb = [int((1 - blend_factor) * c + blend_factor * gray) for c in rgb]
        return '#' + ''.join(f'{c:02x}' for c in blended_rgb)


class StageCTraining(MovingCameraScene):
    def setup(self):
        MovingCameraScene.setup(self)
    def construct(self):
        efficient_net_color = BLUE
        efficient_net_table_color = WHITE


        num_rows = 6
        table_data = [["a"] for _ in range(num_rows)]

        # Add the watermark to the background
        #self.add(addWatermark())

        # Create the table
        efficient_net_table = Table(
            table_data,
            include_outer_lines=True,
            line_config={"color": efficient_net_table_color},
            v_buff = 1.3
        ).shift(LEFT * 5)
        efficient_net_table.get_columns().set_opacity(0)
        efficient_net_table.scale(0.25)

        # Fade in the table
        self.play(FadeIn(efficient_net_table))

        self.wait(2)

        # Encoder
        efficient_net = Polygon(
            [-3.5, -1.25, 0], [-2, -0.5, 0], [-2, 0.5, 0], [-3.5, 1.25, 0],
            color=efficient_net_color, fill_opacity=0
        ).next_to(efficient_net_table, RIGHT)
        efficient_net_text = Text("EfficientNet").scale(0.4).move_to(efficient_net.get_center())
        self.play(FadeIn(efficient_net), Write(efficient_net_text))

        efficient_net_group = VGroup(efficient_net_text, efficient_net, efficient_net_table)

        self.wait(2)
        self.play(efficient_net_group.animate.shift(RIGHT*2))

        self.wait(2)

        database_image = SVGMobject("resources/images/database-solid.svg")
        database_image.scale(0.4).shift(LEFT * 5.5)  # Adjust the scaling and position as needed

        # Display the image
        self.play(FadeIn(database_image))

        # Move the image to the middle of the table and shrink it
        self.wait(2)

        iris_image = ImageMobject("resources/images/iris.jpg")
        iris_image.scale(0.05).move_to(database_image.get_center())

        self.play(iris_image.animate.scale(4))
        database_image_efficient_net_table_arrow = Arrow(database_image.get_right(), efficient_net_table.get_left(), buff=0.1)
        self.play(iris_image.animate.move_to(efficient_net.get_center()).scale(0), GrowArrow(database_image_efficient_net_table_arrow))

        rows, cols, cell_size = 4,4,1
        pixel_grid1 = PixelGrid(rows=rows, cols=cols, cell_size=cell_size)

        # Generate the same colors with more gray (noise)
        colors_with_noise = [[PixelGrid.blend_with_noise(color) for color in row] for row in pixel_grid1.colors]

        # Create the second PixelGrid with the gray colors
        pixel_grid2 = PixelGrid(rows=rows, cols=cols, cell_size=cell_size, colors=colors_with_noise)

        self.add(pixel_grid1)
        pixel_grid1.scale(0.05)
        pixel_grid1.move_to(efficient_net.get_center())
        self.add(pixel_grid1)

        pixel_grid1_dummy = pixel_grid1.copy().scale(4).next_to(efficient_net, RIGHT*3).set_opacity(0)

        efficient_net_pixel_grid1_arrow = Arrow(efficient_net.get_right(), pixel_grid1_dummy.get_left(), buff=0.1)

        self.play(pixel_grid1.animate.scale(4).next_to(efficient_net, RIGHT*3), GrowArrow(efficient_net_pixel_grid1_arrow))

        pixel_grid2.scale(0.2)
        pixel_grid2.next_to(pixel_grid1, RIGHT * 3)

        self.wait(2)
        pixel_grid1_pixel_grid2_arrow = Arrow(pixel_grid1.get_right(), pixel_grid2.get_left(), buff=0.1)
        p1_p2_arrow_text = Text("Noise").scale(0.3)
        self.add(p1_p2_arrow_text.next_to(pixel_grid1_pixel_grid2_arrow, UP*0.5))




        self.play(GrowArrow(pixel_grid1_pixel_grid2_arrow), Write(p1_p2_arrow_text))

        self.play(FadeIn(pixel_grid2))

        self.wait(2)

        equation = MathTex(
            r"X_{sc,t} = \sqrt{\bar{\alpha}_t} \cdot X_{sc} + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon"
        )
        equation.next_to(p1_p2_arrow_text, UP
                         ).scale(0.5)
        self.play(Write(equation))

        self.wait(2)

        self.play(equation.animate.to_edge(UR))

        diffusion_model = Rectangle(width=2, height=1, color=WHITE, fill_opacity=0).next_to(pixel_grid2, RIGHT *3)
        diffusion_model_text = Paragraph('Text-Conditional','Model', alignment='center').scale(0.35).move_to(diffusion_model.get_center())
        self.play(FadeIn(diffusion_model), Write(diffusion_model_text))

        self.wait(2)

        diffusion_model_input_text = Text("Image Caption + timestep").scale(0.3).next_to(diffusion_model, DOWN*3)
        diffusion_model_input_text_arrow = Arrow(diffusion_model_input_text.get_top(), diffusion_model.get_bottom(), buff=0.1)
        self.play(FadeIn(diffusion_model_input_text), GrowArrow(diffusion_model_input_text_arrow))

        self.wait(2)

        pixel_grid2_copy = pixel_grid2.copy()
        p2c_diffusion_model_arrow = Arrow(pixel_grid2.get_right(), diffusion_model.get_left(), buff=0.1)
        self.play(pixel_grid2_copy.animate.move_to(diffusion_model.get_center()).scale(0), GrowArrow(p2c_diffusion_model_arrow))

        pixel_grid1_copy_dummy = pixel_grid1.copy().move_to(diffusion_model.get_center()).scale(0.25).set_opacity(0)
        pixel_grid1_copy_dummy.next_to(diffusion_model, RIGHT*4).scale(4)

        pixel_grid1_copy = pixel_grid1.copy().move_to(diffusion_model.get_center()).scale(0.25)
        diffusion_model_p1c_arrow = Arrow(diffusion_model.get_right(), pixel_grid1_copy_dummy.get_left(), buff=0.1)
        self.play(pixel_grid1_copy.animate.next_to(diffusion_model, RIGHT*4).scale(4), GrowArrow(diffusion_model_p1c_arrow))

        self.wait(2)

        line = Line(start=pixel_grid1.get_top() + UP * 0.5, end=pixel_grid1_copy.get_top() + UP * 0.5, color=WHITE)
        mse_text = Text("Mean Squared Error").scale(0.4).next_to(line, UP * 0.5)

        line_left_vertical = Line(start=line.get_start(), end=line.get_start() + DOWN * 0.5, color=WHITE)
        line_right_vertical = Line(start=line.get_end(), end=line.get_end() + DOWN * 0.5, color=WHITE)

        # Combine the lines
        combined_lines = VGroup(line, line_left_vertical, line_right_vertical)

        self.play(FadeIn(combined_lines), Write(mse_text))

        self.wait(2)

        mse_loss_equation = MathTex(
            r"L_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (\epsilon_i - \hat{\epsilon}_i)^2"
        )

        mse_loss_equation.scale(0.5)
        mse_loss_equation.next_to(mse_text, UP)
        self.play(Write(mse_loss_equation))

        self.wait(2)

        self.play(mse_loss_equation.animate.next_to(equation, DOWN))

        self.wait(2)

        self.camera.frame.save_state()

        # Animation of the camera
        self.play(
            self.camera.auto_zoom(diffusion_model),
            FadeOut(diffusion_model_text, diffusion_model)
        )

        self.wait(2)

        residual_color = BLUE
        cross_attention_color = GREEN
        timestep_color = ORANGE

        # Block dimensions
        block_width = 0.5
        block_height = 2.5

        # Create blocks for the neural network
        example_blocks = VGroup()

        for i in range(3):
            color = residual_color if i % 3 == 0 else cross_attention_color if i % 3 == 1 else timestep_color
            block = Rectangle(width=block_width, height=block_height, fill_color=color, fill_opacity=1, stroke_width=0)
            example_blocks.add(block)

        for i, block in enumerate(example_blocks):
            block.next_to(example_blocks[i - 1], RIGHT, buff=2.25)

        # Annotate specific blocks
        residual_block = example_blocks[0]
        attention_block = example_blocks[1]
        timestep_block = example_blocks[2]

        # Create arrows and labels
        residual_label = Text("Residual Block", font_size=20).next_to(residual_block, DOWN*1)
        attention_label = Paragraph("Attention +", "Cross-Attention Block", font_size=20, alignment='center').next_to(attention_block, DOWN * 1)
        timestep_label = Text("Timestep Block", font_size=20).next_to(timestep_block, DOWN * 1)

        example_setup = VGroup(example_blocks, residual_label, attention_label, timestep_label)
        example_labels = VGroup(residual_label, attention_label, timestep_label)
        example_setup.scale(0.2).move_to(diffusion_model.get_center())

        self.play(FadeIn(example_blocks))
        self.play(Write(residual_label))
        self.play(Write(attention_label))
        self.play(Write(timestep_label))

        self.wait(2)

        self.play(FadeOut(example_labels))

        # Move blocks to the left and next to each other
        target_position = residual_block.get_left() + LEFT*0.2
        animations = []

        for i, block in enumerate(example_blocks):
            if i == 0:
                animations.append(block.animate.move_to(target_position))
            else:
                animations.append(block.animate.next_to(target_position + (i - 1)*RIGHT*0.125, RIGHT, buff=0.075))

        self.play(*animations)

        self.wait(2)

        block_group = VGroup(example_blocks)
        num_copies = 3
        previous_blocks = example_blocks
        for _ in range(num_copies):
            new_blocks = previous_blocks.copy()
            self.play(new_blocks.animate.next_to(previous_blocks, RIGHT, buff=0.075))
            previous_blocks = new_blocks
            block_group.add(new_blocks)

        self.wait(2)
        self.play(Restore(self.camera.frame), block_group.animate.scale(1.3).move_to(diffusion_model.get_center()))

        self.wait(2)


class StageBTraining(MovingCameraScene):
    def setup(self):
        MovingCameraScene.setup(self)

    def construct(self):
        encoder_input_color = WHITE
        encoder_color = GREEN

        num_rows = 8
        table_data = [["a"] for _ in range(num_rows)]

        # Add the watermark to the background
        # self.add(addWatermark())

        # Create the table
        encoder_table = Table(
            table_data,
            include_outer_lines=True,
            line_config={"color": encoder_input_color},
            v_buff=1.3
        ).shift(LEFT * 2)
        encoder_table.get_columns().set_opacity(0)
        encoder_table.scale(0.25)

        # Fade in the table
        self.play(FadeIn(encoder_table))

        self.wait(2)

        # Encoder
        encoder = Polygon(
            [-3.5, -1.5, 0], [-2, -0.5, 0], [-2, 0.5, 0], [-3.5, 1.5, 0],
            color=encoder_color, fill_opacity=0
        )
        encoder.next_to(encoder_table, RIGHT)
        encoder_text = Paragraph("VQGAN","Encoder", alignment='center').scale(0.5).move_to(encoder.get_center())
        self.play(FadeIn(encoder), Write(encoder_text))

        self.wait(2)

        image = ImageMobject("resources/images/iris.jpg")
        image.next_to(encoder_table, LEFT*2).scale(0.4)  # Adjust the scaling and position as needed

        # Display the image
        self.play(FadeIn(image))

        # Move the image to the middle of the table and shrink it
        self.play(image.animate.move_to(encoder.get_center()).scale(0))
        self.remove(image)

        rows, cols, cell_size = 16, 16, 1
        pixel_grid1 = PixelGrid(rows=rows, cols=cols, cell_size=cell_size).scale(0.015)
        pixel_grid1.move_to(encoder.get_center())



        self.play(pixel_grid1.animate.next_to(encoder, RIGHT*4).scale(5))

        noise_count = 3
        pixel_grids = [pixel_grid1.copy()]
        for i in range(noise_count):
            colors_with_noise = [[PixelGrid.blend_with_noise(color, blend_factor=0.35) for color in row] for row in pixel_grids[i].colors]
            pixel_grids.append(PixelGrid(rows=rows, cols=cols, cell_size=cell_size, colors=colors_with_noise).scale(0.075))

        self.wait(2)

        pixel_grid2 = pixel_grids[noise_count].next_to(pixel_grid1, RIGHT*4)
        p1_p2_arrow = Arrow(pixel_grid1.get_right(), pixel_grid2.get_left(), buff=0.1)
        p1_p2_arrow_text = Text("Noise").scale(0.3).next_to(p1_p2_arrow, UP*0.5)
        self.play(GrowArrow(p1_p2_arrow), FadeIn(pixel_grid2), Write(p1_p2_arrow_text))

        self.wait(2)

        vqgan_noise_group = VGroup(pixel_grid1, pixel_grid2, p1_p2_arrow, p1_p2_arrow_text, encoder, encoder_text, encoder_table)
        self.play(vqgan_noise_group.animate.shift(UP*1, LEFT*5).scale(0.8))

        self.wait(2)

        block_width = 0.25
        block_height = 2.5
        block_color1 = BLUE
        block_color2 = TEAL

        # Function to create a block
        def create_block(width, height, color):
            return Rectangle(width=width, height=height, fill_color=color, fill_opacity=1, stroke_width=0)

        # Create downsampling blocks
        down_block1 = VGroup()
        block1 = create_block(block_width, block_height, block_color1)
        block2 = create_block(block_width, block_height, block_color2)

        down_block1.add(block1, block2)
        down_block1.arrange(RIGHT, buff=0.1)
        down_block1.next_to(pixel_grid2, RIGHT, buff=0.5)

        def create_next_block_group(sign, block_group):
            new_block_group = VGroup()

            for block in block_group:
                current_width = block.width
                current_height = block.height

                # Adjust the dimensions based on the sign
                if sign > 0:
                    new_width = current_width + 0.05
                    new_height = current_height - 0.5
                else:
                    new_width = current_width - 0.05
                    new_height = current_height + 0.5

                # Create a new block with the adjusted dimensions
                new_block = create_block(new_width, new_height, block.get_fill_color())
                new_block_group.add(new_block)

            new_block_group.arrange(RIGHT, buff=0.1)
            new_block_group.next_to(block_group, RIGHT, buff=0.25)

            return new_block_group

        down_block2 = create_next_block_group(1, down_block1)
        down_block3 = create_next_block_group(1, down_block2)

        down_blocks = VGroup(down_block1, down_block2, down_block3)

        up_blocks = down_blocks.copy().scale([-1, 1, 1])

        up_blocks.next_to(down_blocks, RIGHT, buff=1)

        # Add all blocks to the scene
        self.play(FadeIn(down_blocks), FadeIn(up_blocks))

        main_line = Line(start=up_blocks[0].get_bottom() + DOWN*0.5, end=down_blocks[0].get_bottom() + DOWN*0.5)

        # Create connecting lines
        connecting_lines = VGroup()
        for block in down_blocks:
            line = Line((block.get_x(),main_line.get_y(), 0), block.get_bottom())
            connecting_lines.add(line)

        for block in up_blocks:
            line = Line((block.get_x(),main_line.get_y(), 0), block.get_bottom())
            connecting_lines.add(line)

        # Add everything to the scene
        self.add(down_blocks, up_blocks, main_line, connecting_lines)

        unet = VGroup(down_blocks, up_blocks)

        self.wait(2)

        path = ArcBetweenPoints(start=up_blocks[0].get_center() + RIGHT, end=pixel_grid2.get_center(), angle=PI)


        for i in range(noise_count):
            pixel_grids[noise_count-1-i].move_to(unet.get_center()).set_opacity(0).scale(0.1)

            # Animate pixel_grid2 disappearing behind the first down block
            self.play(pixel_grids[noise_count - i].animate.move_to(unet.get_center()).fade(1).scale(0))

            # Animate pixel_grid_denoised appearing from the last up block
            self.play(pixel_grids[noise_count-i-1].animate.move_to(up_blocks[0].get_center() + RIGHT).set_opacity(1).scale(10*0.8))

            self.play(MoveAlongPath(pixel_grids[noise_count-i-1], path))


        self.wait(2)

        efficient_net_color = BLUE
        efficient_net_table_color = WHITE

        num_rows = 6
        table_data = [["a"] for _ in range(num_rows)]

        scale_factor_efficient_net = 0.175

        # Create the table
        efficient_net_table = Table(
            table_data,
            include_outer_lines=True,
            line_config={"color": efficient_net_table_color},
            v_buff=1.3
        ).shift(LEFT*6).shift(DOWN*3)
        efficient_net_table.get_columns().set_opacity(0)
        efficient_net_table.scale(scale_factor_efficient_net)

        # Fade in the table
        self.play(FadeIn(efficient_net_table))

        self.wait(2)

        # Encoder
        efficient_net = Polygon(
            [-3.5, -1.25, 0], [-2, -0.5, 0], [-2, 0.5, 0], [-3.5, 1.25, 0],
            color=efficient_net_color, fill_opacity=0
        ).scale(scale_factor_efficient_net/0.25).next_to(efficient_net_table, RIGHT*(scale_factor_efficient_net/0.25))
        efficient_net_text = Text("EfficientNet").scale(0.4*(scale_factor_efficient_net/0.25)).move_to(efficient_net.get_center())
        self.play(FadeIn(efficient_net), Write(efficient_net_text))

        efficient_net_group = VGroup(efficient_net_text, efficient_net, efficient_net_table)

        self.wait(2)

        efficient_net_u_net_line1 = Line(start=efficient_net.get_right(), end=(main_line.get_center()[0], efficient_net.get_y(), 0))

        efficient_net_u_net_line2 = Line(start=efficient_net.get_center()[0], end=(main_line.get_x(), efficient_net.get_y(), 0))

        self.play(FadeIn(efficient_net_u_net_line1))

        self.wait(2)


class UNet(MovingCameraScene):
    def setup(self):
        MovingCameraScene.setup(self)

    def construct(self):

        self.wait(2)

        # Define colors
        conv_color = BLUE
        pool_color = TEAL
        up_color = PURPLE

        # Define positions
        start = LEFT * 5
        end = RIGHT * 5
        step_down = DOWN * 1.5

        # Create convolutional blocks
        conv_blocks = [
            Rectangle(height=1, width=0.2, color=conv_color).move_to(start),
            Rectangle(height=0.8, width=0.2, color=conv_color).move_to(start + step_down),
            Rectangle(height=0.6, width=0.2, color=conv_color).move_to(start + 2 * step_down),
        ]

        # Create downsampling (pooling) layers
        pool_layers = [
            Rectangle(height=1, width=0.2, color=pool_color).next_to(conv_blocks[0], RIGHT, buff=0.5),
            Rectangle(height=0.8, width=0.2, color=pool_color).next_to(conv_blocks[1], RIGHT, buff=0.5),
        ]

        # Create upsampling layers
        up_layers = [
            Rectangle(height=0.6, width=0.2, color=up_color).next_to(conv_blocks[2], RIGHT, buff=0.5),
            Rectangle(height=0.8, width=0.2, color=up_color).next_to(pool_layers[1], RIGHT, buff=0.5),
        ]

        # Create arrows
        arrows = [
            Arrow(start=conv_blocks[0].get_right(), end=pool_layers[0].get_left(), buff=0),
            Arrow(start=pool_layers[0].get_right(), end=conv_blocks[1].get_left(), buff=0),
            Arrow(start=conv_blocks[1].get_right(), end=pool_layers[1].get_left(), buff=0),
            Arrow(start=pool_layers[1].get_right(), end=conv_blocks[2].get_left(), buff=0),
            Arrow(start=conv_blocks[2].get_right(), end=up_layers[0].get_left(), buff=0),
            Arrow(start=up_layers[0].get_right(), end=up_layers[1].get_left(), buff=0),
        ]

        # Add elements to scene
        for block in conv_blocks + pool_layers + up_layers:
            self.play(Create(block))

        for arrow in arrows:
            self.play(Create(arrow))
