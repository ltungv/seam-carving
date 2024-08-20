package main

import "core:fmt"
import "core:image"
import "core:image/png"
import "core:log"
import "core:math"
import "core:mem"
import "core:os"
import "core:slice"
import "core:strconv"
import "core:strings"
import stbi "vendor:stb/image"

Args :: struct {
	input:  string,
	output: string,
	width:  int,
}

Matrix :: struct {
	width:    int,
	height:   int,
	stride:   int,
	elements: []f32,
}

Image :: struct {
	width:    int,
	height:   int,
	stride:   int,
	channels: int,
	pixels:   []f32,
}

main :: proc() {
	logger := log.create_console_logger(.Debug)
	defer log.destroy_console_logger(logger)

	context.logger = logger

	when ODIN_DEBUG {
		tracking_allocator_cleanup :: proc "contextless" (track: ^mem.Tracking_Allocator) {
			if len(track.allocation_map) > 0 {
				log.errorf("=== %v allocations not freed: ===", len(track.allocation_map))
				for _, entry in track.allocation_map {
					log.errorf("- %v bytes @ %v", entry.size, entry.location)
				}
			}
			if len(track.bad_free_array) > 0 {
				log.errorf("=== %v incorrect frees: ===", len(track.bad_free_array))
				for entry in track.bad_free_array {
					log.errorf("- %p @ %v", entry.memory, entry.location)
				}
			}
			mem.tracking_allocator_destroy(track)
		}

		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, context.allocator)
		defer tracking_allocator_cleanup(&track)

		temp_track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&temp_track, context.temp_allocator)
		defer tracking_allocator_cleanup(&temp_track)

		context.allocator = mem.tracking_allocator(&track)
		context.temp_allocator = mem.tracking_allocator(&temp_track)
	}

	args, is_valid_args := parse_args()
	if !is_valid_args {
		log.error("Invalid arguments")
		os.exit(1)
	}
	if !exec(args) {
		log.error("Failed to resize the image")
		os.exit(1)
	}
}

parse_args :: proc() -> (Args, bool) {
	args: Args
	args_idx := 1
	for args_idx < len(os.args) {
		switch os.args[args_idx] {
		case "-i":
			args_idx += 1
			if args_idx >= len(os.args) do return {}, false
			args.input = os.args[args_idx]
		case "-o":
			args_idx += 1
			if args_idx >= len(os.args) do return {}, false
			args.output = os.args[args_idx]
		case "-w":
			args_idx += 1
			if args_idx >= len(os.args) do return {}, false
			width, ok := strconv.parse_int(os.args[args_idx])
			if !ok do return {}, false
			args.width = width
		}
		args_idx += 1
	}
	is_valid_args := len(args.input) > 0 && len(args.output) > 0 && args.width > 0
	return args, is_valid_args
}

exec :: proc(args: Args) -> bool {
	defer free_all(context.temp_allocator)

	img, ok := load_image(args.input)
	if !ok do return false
	defer destroy_image(&img)

	if img.width < args.width {
		log.error("Invalid resize width")
		return false
	}

	grayscale := create_matrix(img.width, img.height)
	defer destroy_matrix(&grayscale)

	gradients := create_matrix(img.width, img.height)
	defer destroy_matrix(&gradients)

	energies := create_matrix(img.width, img.height)
	defer destroy_matrix(&energies)

	seam := make([]int, img.height)
	defer delete(seam)

	convert_image_to_grayscale(grayscale, img)
	calculate_image_gradients(gradients, grayscale)
	iterations := img.width - args.width
	for iter in 0 ..< iterations {
		log.infof("Iteration %v", iter)
		calculate_seam_energies(energies, gradients)
		remove_minimum_seam(seam, &img, &grayscale, &gradients, &energies)
		recalculate_image_gradients(gradients, grayscale, seam)
	}

	return save_image_to_png(img, args.output)
}

destroy_matrix :: proc(mat: ^Matrix) {
	if mat.elements != nil {
		delete(mat.elements)
		mat.elements = nil
	}
}

create_matrix :: proc(width, height: int) -> Matrix {
	mat := Matrix {
		width    = width,
		height   = height,
		stride   = width,
		elements = make([]f32, width * height),
	}
	return mat
}

get_matrix_element :: proc "contextless" (mat: Matrix, idx: [2]int) -> f32 {
	element_idx := idx.y * mat.stride + idx.x
	return mat.elements[element_idx]
}

set_matrix_element :: proc "contextless" (mat: Matrix, idx: [2]int, value: f32) {
	element_idx := idx.y * mat.stride + idx.x
	mat.elements[element_idx] = value
}

destroy_image :: proc(img: ^Image) {
	if img.pixels != nil {
		delete(img.pixels)
		img.pixels = nil
	}
}

create_image :: proc(width, height, channels: int) -> Image {
	img := Image {
		width    = width,
		height   = height,
		channels = channels,
		stride   = width,
		pixels   = make([]f32, width * height * channels),
	}
	return img
}

get_image_pixel :: proc "contextless" (img: Image, idx: [2]int) -> []f32 {
	beg := idx.y * img.stride * img.channels + idx.x * img.channels
	end := beg + img.channels
	return img.pixels[beg:end]
}

set_image_pixel :: proc "contextless" (img: Image, idx: [2]int, value: []f32) {
	beg := idx.y * img.stride * img.channels + idx.x * img.channels
	end := beg + img.channels
	copy(img.pixels[beg:end], value[:img.channels])
}

load_image :: proc(path: string) -> (Image, bool) {
	normalize_pixel :: proc "contextless" (pixel: [$N]$T) -> [N]f32 {
		normalized: [N]f32
		for x, i in pixel {
			normalized[i] = f32(x) / f32(max(T))
		}
		return normalized
	}

	normalize_pixels :: proc "contextless" (normalized: []f32, pixels: [][$N]$T) {
		for pixel, i in pixels {
			for channel, j in normalize_pixel(pixel) {
				normalized[i * N + j] = channel
			}
		}
	}

	img, err := image.load(path)
	if err != nil {
		log.errorf("Failed to load the image: %v", err)
		return {}, false
	}
	defer image.destroy(img)

	if !image.is_valid_image(img) {
		log.error("Invalid image format")
		return {}, false
	}

	if !image.alpha_drop_if_present(img, {.alpha_premultiply, .blend_background}) {
		log.error("Failed to drop the alpha channel")
		return {}, false
	}

	result := create_image(img.width, img.height, img.channels)
	switch img.channels {
	case 1:
		switch img.depth {
		case 8:
			normalize_pixels(result.pixels, mem.slice_data_cast([]image.G_Pixel, img.pixels.buf[:]))
		case 16:
			normalize_pixels(result.pixels, mem.slice_data_cast([]image.G_Pixel_16, img.pixels.buf[:]))
		case:
			unreachable()
		}
	case 3:
		switch img.depth {
		case 8:
			normalize_pixels(result.pixels, mem.slice_data_cast([]image.RGB_Pixel, img.pixels.buf[:]))
		case 16:
			normalize_pixels(result.pixels, mem.slice_data_cast([]image.RGB_Pixel_16, img.pixels.buf[:]))
		case:
			unreachable()
		}
	case:
		unreachable()
	}
	return result, true
}

convert_image_to_grayscale :: proc(dst: Matrix, src: Image) {
	convert_pixel_to_grayscale :: proc "contextless" (pixel: [3]f32) -> f32 {
		linearized: [3]f32
		for x, i in pixel {
			if x <= 0.04045 {
				linearized[i] = x / 12.92
			} else {
				linearized[i] = math.pow((x + 0.055) / 1.055, 2.4)
			}
		}
		luminance := 0.2126 * linearized.r + 0.7152 * linearized.g + 0.0722 * linearized.b
		lightness: f32
		if luminance <= 216.0 / 24389.0 {
			lightness = luminance * (24389.0 / 27.0)
		} else {
			lightness = math.pow(luminance, 1.0 / 3.0) * 116.0 - 16.0
		}
		return lightness / 100.0
	}

	assert(dst.width == src.width)
	assert(dst.height == src.height)
	switch src.channels {
	case 1:
		copy(dst.elements, src.pixels)
	case 3:
		for row in 0 ..< dst.height {
			for col in 0 ..< dst.width {
				pixel := get_image_pixel(src, {col, row})
				grayscale := convert_pixel_to_grayscale({pixel[0], pixel[1], pixel[2]})
				set_matrix_element(dst, {col, row}, grayscale)
			}
		}
	case:
		unreachable()
	}
}

calculate_pixel_gradients :: proc "contextless" (gradients: Matrix, grayscale: Matrix, index: [2]int) {
	hfilter := [?]f32{1, 2, 1, 0, 0, 0, -1, -2, -1}
	vfilter := [?]f32{1, 0, -1, 2, 0, -2, 1, 0, -1}
	hgradient: f32 = 0
	vgradient: f32 = 0
	for filter_row in 0 ..< 3 {
		pixel_row := index[1] + filter_row - 1
		if pixel_row < 0 || pixel_row >= grayscale.height do continue
		for filter_col in 0 ..< 3 {
			pixel_col := index[0] + filter_col - 1
			if pixel_col < 0 || pixel_col >= grayscale.width do continue
			gray_level := get_matrix_element(grayscale, {pixel_col, pixel_row})
			filter_idx := filter_row * 3 + filter_col
			hgradient += gray_level * hfilter[filter_idx]
			vgradient += gray_level * vfilter[filter_idx]
		}
	}
	gradient := math.sqrt(hgradient * hgradient + vgradient * vgradient)
	set_matrix_element(gradients, index, gradient)
}

calculate_image_gradients :: proc(gradients: Matrix, grayscale: Matrix) {
	assert(gradients.width == grayscale.width)
	assert(gradients.height == grayscale.height)
	for row in 0 ..< grayscale.height {
		for col in 0 ..< grayscale.width {
			calculate_pixel_gradients(gradients, grayscale, {col, row})
		}
	}
}

recalculate_image_gradients :: proc(gradients: Matrix, grayscale: Matrix, seam: []int) {
	assert(gradients.width == grayscale.width)
	assert(gradients.height == grayscale.height)
	assert(gradients.height == len(seam))
	for col, row in seam {
		for nearby_col in col - 2 ..< col + 2 {
			if nearby_col < 0 || nearby_col >= gradients.width do continue
			calculate_pixel_gradients(gradients, grayscale, {nearby_col, row})
		}
	}
}

calculate_seam_energies :: proc(energies: Matrix, gradients: Matrix) {
	assert(energies.width == gradients.width)
	assert(energies.height == gradients.height)
	for col in 0 ..< energies.width {
		gradient := get_matrix_element(gradients, {col, 0})
		set_matrix_element(energies, {col, 0}, gradient)
	}
	for row in 1 ..< energies.height {
		for col in 0 ..< energies.width {
			min_energy := max(f32)
			for prev_col in col - 1 ..= col + 1 {
				if prev_col < 0 || prev_col >= energies.width do continue
				energy := get_matrix_element(energies, {prev_col, row - 1})
				min_energy = min(min_energy, energy)
			}
			gradient := get_matrix_element(gradients, {col, row})
			set_matrix_element(energies, {col, row}, min_energy + gradient)
		}
	}
}

remove_minimum_seam :: proc(seam: []int, img: ^Image, gray: ^Matrix, grad: ^Matrix, energies: ^Matrix) {
	remove_row_minimum :: proc "contextless" (
		img: ^Image,
		gray: ^Matrix,
		grad: ^Matrix,
		energies: Matrix,
		row: int,
		col_beg: int,
		col_end: int,
	) -> int {
		min_col: int
		min_energy := max(f32)
		for col in col_beg ..= col_end {
			if col < 0 || col >= img.width do continue
			energy := get_matrix_element(energies, {col, row})
			if energy < min_energy {
				min_col = col
				min_energy = energy
			}
		}
		row_offset := row * img.stride
		row_curr_idx := row_offset + min_col
		row_last_idx := row_offset + img.width
		img_row_curr_idx := row_curr_idx * img.channels
		img_row_last_idx := row_last_idx * img.channels
		copy(gray.elements[row_curr_idx:row_last_idx - 1], gray.elements[row_curr_idx + 1:row_last_idx])
		copy(grad.elements[row_curr_idx:row_last_idx - 1], grad.elements[row_curr_idx + 1:row_last_idx])
		copy(
			img.pixels[img_row_curr_idx:img_row_last_idx - img.channels],
			img.pixels[img_row_curr_idx + img.channels:img_row_last_idx],
		)
		return min_col
	}

	assert(img.width == gray.width)
	assert(img.width == grad.width)
	assert(img.width == energies.width)
	assert(img.height == gray.height)
	assert(img.height == grad.height)
	assert(img.height == energies.height)
	assert(img.height == len(seam))

	row := img.height - 1
	col := remove_row_minimum(img, gray, grad, energies^, row, 0, img.width - 1)
	seam[row] = col
	for row > 0 {
		row -= 1
		col = remove_row_minimum(img, gray, grad, energies^, row, col - 1, col + 1)
		seam[row] = col
	}
	energies.width -= 1
	gray.width -= 1
	grad.width -= 1
	img.width -= 1
}

save_matrix_to_png :: proc(mat: Matrix, fname: string) -> bool {
	find_matrix_min_max :: proc "contextless" (mat: Matrix) -> (f32, f32) {
		min_element := max(f32)
		max_element := min(f32)
		for row in 0 ..< mat.height {
			for col in 0 ..< mat.width {
				element := get_matrix_element(mat, {col, row})
				min_element = min(min_element, element)
				max_element = max(max_element, element)
			}
		}
		return min_element, max_element
	}

	buf := make([]u8, mat.width * mat.height)
	defer delete(buf)

	min_element, max_element := find_matrix_min_max(mat)
	for row in 0 ..< mat.height {
		for col in 0 ..< mat.width {
			idx := row * mat.width + col
			elem := get_matrix_element(mat, {col, row})
			norm := (elem - min_element) / (max_element - min_element)
			buf[idx] = u8(norm * 255)
		}
	}

	fname_cstring := strings.clone_to_cstring(fname)
	defer delete(fname_cstring)

	write_err := stbi.write_png(fname_cstring, i32(mat.width), i32(mat.height), 1, slice.as_ptr(buf), i32(mat.width))
	return write_err != 0
}

save_image_to_png :: proc(img: Image, fname: string) -> bool {
	buf := make([]u8, img.width * img.height * img.channels)
	defer delete(buf)

	for row in 0 ..< img.height {
		for col in 0 ..< img.width {
			idx := row * img.width * img.channels + col * img.channels
			for c, i in get_image_pixel(img, {col, row}) {
				buf[idx + i] = u8(c * 255)
			}
		}
	}

	fname_cstring := strings.clone_to_cstring(fname)
	defer delete(fname_cstring)

	write_err := stbi.write_png(
		fname_cstring,
		i32(img.width),
		i32(img.height),
		i32(img.channels),
		slice.as_ptr(buf),
		i32(img.width) * i32(img.channels),
	)
	return write_err != 0
}
