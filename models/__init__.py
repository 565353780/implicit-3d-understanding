from . import total3d, mgnet, ldif

method_paths = {
    'TOTAL3D': total3d,
    'MGNet': mgnet,
    'LDIF': ldif
}

__all__ = ['method_paths']
